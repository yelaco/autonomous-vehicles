#include <opencv2/opencv.hpp>
#include <opencv2/tracking/tracking.hpp>
#include <string>
#include <fstream>
#include <cstring>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <thread>
#include <atomic>
#include <string>

std::vector<std::string> load_class_list()
{
    std::vector<std::string> class_list;
    std::ifstream ifs("config_files/classes.txt");
    std::string line;
    while (getline(ifs, line))
    {
        class_list.push_back(line);
    }
    return class_list;
}

void load_net(cv::dnn::Net &net, bool is_cuda)
{
    auto result = cv::dnn::readNet("config_files/best1404.onnx");
    if (is_cuda)
    {
        std::cout << "Attempty to use CUDA\n";
        result.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
        result.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA_FP16);
    }
    else
    {
        std::cout << "Running on CPU\n";
        result.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
        result.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
    }
    net = result;
}

cv::Ptr<cv::Tracker> init_tracker(cv::Mat frame, cv::Rect2d bbox) {
    cv::Ptr<cv::Tracker> tracker = cv::TrackerCSRT::create();
    tracker->init(frame, bbox);
    return tracker;
}

std::string get_decision(cv::Rect bbox, double width) {
    double cx = bbox.x + bbox.width / 2;

    // Calculate slope
    if (cx < width / 2 - 50)
        return "Turn left";
    else if (cx > width / 2 + 50) 
        return "Turn right";
    else  
        return "Go straight";
}

int count_red_pixels(cv::Mat image) {
    // Convert image to HSV color space
    cv::Mat hsv;
    cv::cvtColor(image, hsv, cv::COLOR_BGR2HSV);
    std::vector<cv::Mat> channels;
    cv::split(hsv, channels);
    cv::Mat v_eq;
    cv::equalizeHist(channels[2], v_eq);
    cv::merge(std::vector<cv::Mat>{channels[0], channels[1], v_eq}, hsv);
    
    // Define lower and upper bounds for red color
    cv::Scalar lower_red(0, 100, 100);
    cv::Scalar upper_red(10, 255, 255);

    // Mask out red pixels
    cv::Mat mask1;
    cv::inRange(hsv, lower_red, upper_red, mask1);

    // Define lower and upper bounds for slightly different red color
    cv::Scalar lower_red2(160, 100, 100);
    cv::Scalar upper_red2(180, 255, 255);

    // Mask out the second range of red pixels
    cv::Mat mask2;
    cv::inRange(hsv, lower_red2, upper_red2, mask2);

    // Combine masks
    cv::Mat mask;
    cv::bitwise_or(mask1, mask2, mask);

    // Count red pixels
    int red_pixel_count = cv::countNonZero(mask);

    return red_pixel_count;
}

const float INPUT_WIDTH = 640.0;
const float INPUT_HEIGHT = 640.0;
const float SCORE_THRESHOLD = 0.25;
const float NMS_THRESHOLD = 0.45;
const float CONFIDENCE_THRESHOLD = 0.4;

struct Detection
{
    int class_id;
    float confidence;
    cv::Rect box;
};

cv::Mat format_yolov5(const cv::Mat &source) {
    int col = source.cols;
    int row = source.rows;
    int _max = MAX(col, row);
    cv::Mat result = cv::Mat::zeros(_max, _max, CV_8UC3);
    source.copyTo(result(cv::Rect(0, 0, col, row)));
    return result;
}

void detect(cv::Mat &image, cv::dnn::Net &net, std::vector<Detection> &output, const std::vector<std::string> &className) {
    cv::Mat blob;

    auto input_image = format_yolov5(image);
    
    cv::dnn::blobFromImage(input_image, blob, 1./255., cv::Size(INPUT_WIDTH, INPUT_HEIGHT), cv::Scalar(), true, false);
    net.setInput(blob);
    std::vector<cv::Mat> outputs;
    net.forward(outputs, net.getUnconnectedOutLayersNames());

    float x_factor = input_image.cols / INPUT_WIDTH;
    float y_factor = input_image.rows / INPUT_HEIGHT;
    
    float *data = (float *)outputs[0].data;

    const int dimensions = className.size() + 5;
    const int rows = outputs[0].size().width;
    
    std::vector<int> class_ids;
    std::vector<float> confidences;
    std::vector<cv::Rect> boxes;

    for (int i = 0; i < rows; ++i) {
        float confidence = data[4];
        if (confidence >= CONFIDENCE_THRESHOLD) {
            float * classes_scores = data + dimensions;
            cv::Mat scores(1, className.size(), CV_32FC1, classes_scores);
            cv::Point class_id;
            double max_class_score;
            minMaxLoc(scores, 0, &max_class_score, 0, &class_id);
            if (max_class_score >= SCORE_THRESHOLD) {
                confidences.push_back(confidence);
                class_ids.push_back(class_id.x);
                float x = data[0];
                float y = data[1];
                float w = data[2];
                float h = data[3];
                int left = int((x - 0.5 * w) * x_factor);
                int top = int((y - 0.5 * h) * y_factor);
                int width = int(w * x_factor);
                int height = int(h * y_factor);
                boxes.push_back(cv::Rect(left, top, width, height));
            }
        }

        data += dimensions;
    }

    if (boxes.size() > 0) {
        std::vector<int> nms_result;
        cv::dnn::NMSBoxes(boxes, confidences, SCORE_THRESHOLD, NMS_THRESHOLD, nms_result);
        for (int i = 0; i < nms_result.size(); i++) {
            int idx = nms_result[i];
            Detection result;
            result.class_id = class_ids[idx];
            result.confidence = confidences[idx];
            result.box = boxes[idx];
            output.push_back(result);
        }
    }
}

void send_data(int sockfd, std::string data) {
    if (send(sockfd, data.c_str(), data.length(), 0) < 0) {
        std::cerr << "Error sending data" << std::endl;
        exit(1);
    }
}

class DetectBufferThread {
public:
    DetectBufferThread(cv::dnn::Net net, std::vector<std::string> className)
        : net(net), className(className), detecting(false) {
        thread = std::thread(&DetectBufferThread::run, this);
    }

    ~DetectBufferThread() {
        detecting = false;
        if (thread.joinable()) {
            thread.join();
        }
    }

    void start_detect(cv::Mat frame) {
        this->detecting = true;
        this->last_frame = frame;
    }

    bool is_detecting() {
        return this->detecting;
    }

    std::vector<Detection> get_latest_output() {
        if (this->outputs.empty()) {
            return std::vector<Detection>();
        }
        auto output = outputs.back();
        outputs.clear();
        return output;
    }

private:
    std::atomic<bool> detecting;
    std::thread thread;
    cv::dnn::Net net;
    std::vector<std::vector<Detection>> outputs;
    std::vector<std::string> className;
    cv::Mat last_frame;

    void run() {
        while (true) {
            if (detecting) {
                std::vector<Detection> output;
                detect(this->last_frame, this->net, output, this->className);
                this->outputs.push_back(output);
                detecting = false;
            }
        }
    }
};

int main(int argc, char**argv)
{
    std::cout << "Enter pi's ip address: ";
    std::string server_ip{};
    std::getline(std::cin, server_ip);
    int port{65432};
    std::vector<std::string> class_list = load_class_list();
    
    auto net = cv::dnn::readNet("config_files/best1404.onnx");

    int sockfd = socket(AF_INET, SOCK_STREAM, 0);
    if (sockfd < 0) {
        std::cerr << "Error creating socket" << std::endl;
        return 1;
    }

    // Server address structure
    struct sockaddr_in serverAddr;
    memset(&serverAddr, 0, sizeof(serverAddr));
    serverAddr.sin_family = AF_INET;
    serverAddr.sin_port = htons(port); // Port number

    // Convert IP address from presentation format to network format
    if (inet_pton(AF_INET, server_ip.c_str(), &serverAddr.sin_addr) <= 0) {
        std::cerr << "Invalid address/ Address not supported" << std::endl;
        return 1;
    }

    // Connect to the server
    if (connect(sockfd, (const sockaddr*)&serverAddr, sizeof(serverAddr)) < 0) {
        std::cerr << "Connection failed" << std::endl;
        return 1;
    }

    bool is_tracking = false;
    cv::Ptr<cv::Tracker> tracker{};
    cv::Rect bbox{};
    std::string decision{"None"};
    std::string obj_label{};
    float conf{-1.0};
    int stop_threshold = 120000;

    cv::Mat frame;
    cv::namedWindow("Autonomous Vehicle");
    cv::VideoCapture cap("rtsp://" + server_ip + ":8554/video_stream");
    if (!cap.isOpened()) {
        std::cout << "No video stream detected" << '\n';
        system("pause");
        return -1;
    }
    // cap.set(cv::CAP_PROP_BUFFERSIZE, 5);

    double vid_height = cap.get(cv::CAP_PROP_FRAME_HEIGHT);
    double vid_width = cap.get(cv::CAP_PROP_FRAME_WIDTH);
    float fps{-1};

    DetectBufferThread detect_buffer(net, class_list);
    
    while (true && "Stop" != decision)
    {
        cap >> frame;

        if (frame.empty()){ //Breaking the loop if no video frame is detected//
            break;
        }

        auto timer = cv::getTickCount();

        int num_red_pixel = count_red_pixels(frame);
        if (num_red_pixel > stop_threshold) {
            decision = "Stop";
            is_tracking = false;
            send_data(sockfd, decision); 
        } else {
            std::cout << num_red_pixel << '\n'; 
        }

        if (is_tracking) 
        {
            if (tracker->update(frame, bbox)) 
            {
                int cx = int(bbox.x + bbox.width / 2);
                int cy = int(bbox.y + bbox.height / 2);
                cv::line(frame, cv::Point(cx, cy), cv::Point(vid_width / 2, vid_height), cv::Scalar(255,255,255), 2);

                decision = get_decision(bbox, vid_width);
                  
                send_data(sockfd, decision); 

                std::ostringstream text;
                text << obj_label << " " << std::fixed << std::setprecision(2) << conf;
                cv::rectangle(frame, cv::Point(bbox.x, bbox.y), cv::Point(bbox.x + bbox.width, bbox.y + bbox.height), cv::Scalar(255,0,0), 2);
                cv::putText(frame, text.str(), cv::Point(bbox.x,bbox.y-2),cv::FONT_HERSHEY_COMPLEX, 0.7,cv::Scalar(255,0,255),2);
                cv::putText(frame, decision, cv::Point(160, 80), cv::FONT_HERSHEY_SIMPLEX, 2, cv::Scalar(0, 0, 255), 2); 
            } else {
                is_tracking = false;
            }
        }

        std::vector<Detection> output = detect_buffer.get_latest_output();
        if (output.empty()) {
            if (!detect_buffer.is_detecting()) 
                detect_buffer.start_detect(frame);
        } else {
            int parking_id{-1};
            float max_conf{-1.0};
            for (int i = 0; i < output.size(); ++i)
            {
                auto detection = output[i];
                if (class_list[detection.class_id] == "parking" && max_conf < detection.confidence)
                {
                    parking_id = i;
                    max_conf = detection.confidence;
                } 
            }
            if (parking_id > -1) {
                bbox = output[parking_id].box;
                obj_label = class_list[output[parking_id].class_id];
                conf = max_conf;
                tracker = init_tracker(frame, bbox);
                is_tracking = true; 
                send_data(sockfd, decision); 
            }
        }
        cv::putText(frame, "FPS: " + std::to_string((int) (cv::getTickFrequency() / (cv::getTickCount() - timer))),cv::Point(10, 25), cv::FONT_HERSHEY_SIMPLEX, 0.75, cv::Scalar(50,170,50), 2);

        cv::imshow("Autonomous Vehicle", frame);

        char c = (char) cv::waitKey(1);
        if (c == 27){ //If 'Esc' is entered break the loop//
            break;
        }
    }

    cap.release();
    close(sockfd);
    std::cout << "Finished" << '\n';

    return 0;
}

#include <opencv2/opencv.hpp>
#include <opencv2/tracking/tracking.hpp>
#include <string>
#include <fstream>
#include <cstring>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>

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

std::string get_decision(cv::Rect bbox) {
    return "Decision";
}

const std::vector<cv::Scalar> colors = {cv::Scalar(255, 255, 0), cv::Scalar(0, 255, 0), cv::Scalar(0, 255, 255), cv::Scalar(255, 0, 0)};

const float INPUT_WIDTH = 640.0;
const float INPUT_HEIGHT = 640.0;
const float SCORE_THRESHOLD = 0.2;
const float NMS_THRESHOLD = 0.4;
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

    const int dimensions = 85;
    const int rows = 25200;
    
    std::vector<int> class_ids;
    std::vector<float> confidences;
    std::vector<cv::Rect> boxes;

    for (int i = 0; i < rows; ++i) {

        float confidence = data[4];
        if (confidence >= CONFIDENCE_THRESHOLD) {

            float * classes_scores = data + 5;
            cv::Mat scores(1, className.size(), CV_32FC1, classes_scores);
            cv::Point class_id;
            double max_class_score;
            minMaxLoc(scores, 0, &max_class_score, 0, &class_id);
            if (max_class_score > SCORE_THRESHOLD) {

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

        data += 85;
    }

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

int main(int argc, char**argv)
{
    std::cout << "Enter pi's ip address: ";
    std::string server_ip{};
    std::getline(std::cin, server_ip);
    int port{65432};
    std::vector<std::string> class_list = load_class_list();

    bool is_cuda = argc > 1 && strcmp(argv[1], "cuda") == 0;
    
    cv::dnn::Net net;
    load_net(net, is_cuda);

    int sockfd = socket(AF_INET, SOCK_STREAM, 0);
    if (sockfd < 0) {
        std::cerr << "Error creating socket" << std::endl;
        return 1;
    }

    // Server address structure
    struct sockaddr_in serverAddr;
    memset(&serverAddr, 0, sizeof(serverAddr));
    serverAddr.sin_family = AF_INET;
    serverAddr.sin_port = htons(8080); // Port number

    // Convert IP address from presentation format to network format
    if (inet_pton(AF_INET, server_ip.c_str(), &serverAddr.sin_addr) <= 0) {
        std::cerr << "Invalid address/ Address not supported" << std::endl;
        return 1;
    }

    // Connect to the server
    // if (connect(sockfd, (const sockaddr*)&serverAddr, sizeof(serverAddr)) < 0) {
    //     std::cerr << "Connection failed" << std::endl;
    //     return 1;
    // }

    cv::Mat frame;
    cv::VideoCapture cap("rtsp://" + server_ip + ":8554/video_stream?dummy=test.mjpg", cv::CAP_FFMPEG);
    
    bool is_tracking = false;
    cv::Ptr<cv::Tracker> tracker{};
    cv::Rect bbox{};
    std::string decision{"None"};
    std::string obj_label{};
    int conf{-1};

    while (true)
    {
        cap >> frame;
        if (frame.empty()) {
            continue; // to be replaced with cam cleaner later
        }

        cv::imshow("frame", frame);
        continue;
        if (is_tracking) 
        {
            if (tracker->update(frame, bbox)) 
            {
                int cx = int(bbox.x + bbox.width / 2);
                int cy = int(bbox.y + bbox.height / 2);
                cv::line(frame, cv::Point(cx, cy), cv::Point(INPUT_WIDTH / 2, INPUT_HEIGHT), (255,255,255), 2);

                decision = get_decision(bbox);
                
                // if (send(sockfd, decision.c_str(), decision.length(), 0) < 0) {
                //     std::cerr << "Error sending data" << std::endl;
                //     return 1;
                // }

                std::string text = obj_label;
                cv::rectangle(frame, cv::Point(bbox.x, bbox.y), cv::Point(bbox.x + bbox.width, bbox.y + bbox.height), (255,0,0), 2);
                cv::putText(frame, text, cv::Point(bbox.x,bbox.y-2),cv::FONT_HERSHEY_COMPLEX, 0.7,(255,0,255),2);
                cv::putText(frame, decision, cv::Point(160, 80), cv::FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2); 
            } else {
                is_tracking = false;
            }
        }
        std::vector<Detection> output;
        detect(frame, net, output, class_list);
        
        int detections = output.size();

        for (int i = 0; i < detections; ++i)
        {
            auto detection = output[i];
            auto bbox = detection.box;
            auto classId = detection.class_id;
            if (class_list[detection.class_id] == "parking")
            {
                const auto color = colors[classId % colors.size()];
                // add overlay and init tracking
                tracker = init_tracker(frame, bbox);
                is_tracking = true;
                break;
            }
        }

        cv::imshow("output", frame);

        if (cv::waitKey(1) != -1)
        {
            cap.release();
            std::cout << "finished by user\n";
            break;
        }
    }

    // close(sockfd);
    
    return 0;
}

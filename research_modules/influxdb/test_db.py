import influxdb_client, os, time
from influxdb_client import InfluxDBClient, Point, WritePrecision
from influxdb_client.client.write_api import SYNCHRONOUS

token = "URd33pD5M87SL74YJ3QPhvAyRi6twIh9_5D-UCM2rR_UMOz7VOlVNvCUaGo8dCgHjJMCuJWQUVIueu9zXa_-rQ=="
org = "ivan"
url = "http://35.240.198.118:8086"

client = influxdb_client.InfluxDBClient(url=url, token=token, org=org)

bucket="navy-ivan"

write_api = client.write_api(write_options=SYNCHRONOUS)
   
for i in range(4):
  point = (
    Point("vehicle_sensors")
    .tag("vehicle_id", f"{i}")
    .field("hc-sr04", "[100, 30, 40, 50, 60]")
    .field("lidar", "lidar_test")
    .field("gps", "test")
  )
  write_api.write(bucket=bucket, org="ivan", record=point)
  time.sleep(1) # separate points by 1 second

query_api = client.query_api()

query = """from(bucket: "navy-ivan")
 |> range(start: -10m)
 |> filter(fn: (r) => r._measurement == "vehicle_sensors")"""
tables = query_api.query(query, org="ivan")

for table in tables:
  for record in table.records:
    print(record)
    
# query_api = client.query_api()

# query = """from(bucket: "navy-ivan")
#   |> range(start: -10m)
#   |> filter(fn: (r) => r._measurement == "measurement1")
#   |> mean()"""
# tables = query_api.query(query, org="ivan")

# for table in tables:
#     for record in table.records:
#         print(record)
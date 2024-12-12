#include "yolo-fastestv2.h"
#include <iostream>
#include <vector>
#include <nlohmann/json.hpp> // JSON库

using json = nlohmann::json;

int main() {
    static const char* class_names[] = {
        "face"
    };

    yoloFastestv2 api;
    api.loadModel("./yolo_fastestv2/model/yolo_fastestv2_256-opt.param", "./yolo_fastestv2/model/yolo_fastestv2_256-opt.bin");

    while (true) {
        // 读取图像大小
        uint32_t img_size = 0;
        std::cin.read(reinterpret_cast<char*>(&img_size), sizeof(img_size));
        if (img_size == 0) break; // 结束信号

        // 读取图像数据
        std::vector<uchar> img_data(img_size);
        std::cin.read(reinterpret_cast<char*>(img_data.data()), img_size);

        // 解码图像
        cv::Mat cvImg = cv::imdecode(img_data, cv::IMREAD_COLOR);
        if (cvImg.empty()) {
            std::cerr << "Error: Failed to decode image" << std::endl;
            std::cout.flush();
            continue;
        }

        // 检测
        std::vector<TargetBox> boxes;
        api.detection(cvImg, boxes);

        // 构造JSON结果
        json result = json::array();
        for (const auto& box : boxes) {
            result.push_back({
                {"x1", box.x1},
                {"y1", box.y1},
                {"x2", box.x2},
                {"y2", box.y2},
                {"score", box.score},
                {"class", class_names[box.cate]}
            });
        }

        // 输出结果到stdout
        std::cout << result.dump() << std::endl;
        std::cout.flush();
    }

    return 0;
}

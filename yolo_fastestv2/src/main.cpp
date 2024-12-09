#include "yolo-fastestv2.h"
#include <iostream>
#include <vector>
#include <nlohmann/json.hpp> // JSON库

using json = nlohmann::json;

int main() {
    static const char* class_names[] = {
        "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
        "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
        "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
        "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
        "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
        "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
        "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
        "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
        "hair drier", "toothbrush"
    };

    yoloFastestv2 api;
    api.loadModel("./yolo_fastestv2/model/yolo-fastestv2-opt.param", "./yolo_fastestv2/model/yolo-fastestv2-opt.bin");

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

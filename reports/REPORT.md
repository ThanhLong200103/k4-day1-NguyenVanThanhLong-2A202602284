# Báo cáo bài thực hành Ngày 1 – Đọc nhãn từ đầu ra YOLO11

**Ngày chạy:11/9/2026

**Runtime Colab:** CPU

**Python / PyTorch / Ultralytics:8.4.145

**Checkpoint:** /
yolo11n-cls.pt:https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo11n-cls.pt,
yolo11n.pt:https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo11n.pt,
yolo11n-seg.pt:https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo11n-seg.pt

**Thay đổi so với notebook nguồn:** Không / mô tả rõ thay đổi

> ZIP do notebook tạo có tên `<KHOA>-DAY01-report.zip` (ví dụ: `K4-DAY01-report.zip`). Giải nén rồi đặt trực tiếp `REPORT.md` và
> `day1_lab_outputs/` vào thư mục `report/` của repository tạo từ template. Không ghi họ tên, MSSV,
> email, số điện thoại hoặc dữ liệu cá nhân khác. Nộp link repository trên VLearn; tài khoản VLearn xác
> định người nộp.

## 1. Phân loại ảnh – prediction cấp ảnh

Nguồn evidence: `classification_predictions.json`, sample `traffic`.

- Record hạng 1 (`class_id`, `class_name`, `rank`, `score`, `taxonomy_name`):
+ "class_id": 468,
+ "class_name": "cab",
+ "rank": 1,
+  "score": 0.510915
+ "taxonomy_name": "ImageNet-1K",
- Record này mô tả toàn ảnh như thế nào?
+ mô tả mức độ phân loại xe , chiều cao chi tiết từng xe , và xếp hạng hộ chính xác cũng như số lượng của từng loại xe , lấy ra top5 class_id có điểm cao nhất

- Ai định nghĩa class list mà checkpoint có thể dự đoán?
Class list được định nghĩa bởi dataset và cấu hình nhãn dùng để huấn luyện checkpoint

- Vì sao cần giữ cả ID, tên lớp và tên taxonomy?
id dùng để định danh thuận tiên cho máy tính xử lý , tên để con người đọc , còn taxonomy để phân loại trên hệ thống
- Nếu ảnh có nhiều chủ thể, guideline cần quy định điều gì?
Nhiệm vụ là gán một nhãn cho toàn ảnh hay xác định nhãn cho từng chủ thể.
Nếu có nhiều chủ thể khác nhau, chọn chủ thể chính, nhãn chi phối hay cho phép nhiều nhãn.
Cách xử lý chủ thể nhỏ, bị che khuất, ở xa hoặc không đủ thông tin.
Khi ảnh không phù hợp với taxonomy hoặc không thể quyết định, annotator phải chọn unknown, ambiguous hoặc chuyển reviewer.
- Vì sao model score không phải ground truth?
score model nó chỉ là độ tin tưởng dự đoán đến mức nào chứ không chứng minh được nó đúng như ground truth do phải annotator hoặc con người review

## 2. Phát hiện vật thể – lớp và box cho từng object

Nguồn evidence: `detection_predictions.json` và `visuals/detection_predictions.png`, sample `kitchen`.

- Một record (`class_name`, `score`, `bbox_xyxy`, `bbox_width`, `bbox_height`):
+ "class_name": "bus",
+ "score": 0.912557,
+ "bbox_xyxy": [
      93.17,
      187.95,
      223.01,
      320.91
    ],
+ "bbox_width": 129.84,
+ "bbox_height": 132.96
- Diễn giải vị trí box bằng lời:
xe buýt nằm ở khu vực bên trái, hơi thấp hơn trung tâm ảnh, chiếm một vùng hình chữ nhật có chiều rộng và chiều cao gần tương đương. Model nhận diện là bus với score 0.912557, vượt ngưỡng dự đoán 0.35.

- So sánh số prediction ở hai threshold:
Khi threshold=0.20, model thường trả về nhiều prediction nhất. Khi tăng lên 0.35, các prediction có score thấp bị loại bỏ nên số lượng giảm. Ở 0.60, chỉ giữ lại các prediction có độ tin cậy cao nhất nên số lượng thường thấp nhất. Số cụ thể được lấy từ kết 
- Điều gì thay đổi đối với độ bao phủ và khối lượng reviewer cần xem?
threshold thấp tăng mức độ bao phủ , giảm nguy cơ bỏ sót vật thể  những tạo nhiều fasle positive , khiến reviewer phải kiểm tra nhiều hơn. Threshold cao làm giảm số prediction và giảm khối lượng review, nhưng có thể bỏ sót các vật thể nhỏ, mờ, bị che khuất hoặc có score thấp. Threshold 0.35 là mức cân bằng giữa độ bao phủ và
- Đề xuất một quy tắc box chặt:Box phải bao quanh toàn bộ phần nhìn thấy của đúng một object, sát biên object nhất có thể, không bao gồm quá nhiều nền và không gộp nhiều object khác nhau vào cùng một box. Với object bị che khuất, box chỉ bao quanh phần có thể xác định được của object, theo đúng guideline của dự án.
- Với object bị che khuất/cắt mép, điều gì cần guideline hoặc escalation quyết định?
Guideline cần quy định có gán nhãn khi chỉ nhìn thấy một phần object hay không, box bao quanh phần nhìn thấy hay ước lượng toàn bộ object, và tỷ lệ nhìn thấy tối thiểu để chấp nhận. Nếu object bị che quá nhiều, cắt khỏi mép ảnh, không xác định rõ lớp hoặc nhiều object chồng lấn, annotator nên đánh dấu
## 3. Phân đoạn theo từng đối tượng – polygon cho mỗi instance

Nguồn evidence: `segmentation_predictions.json` và `visuals/segmentation_prediction.png`, sample `kitchen`.

- Một record (`instance_id`, `class_name`, `score`, số điểm và một phần `polygon_xy`):
{
    "sample_id": "traffic",
    "coco_image_id": 210273,
    "image_width": 640,
    "image_height": 428,
    "task": "instance_segmentation",
    "taxonomy_name": "COCO-80",
    "model_file": "yolo11n-seg.pt",
    "model_sha256": "55ed65c56c91713d23e8402371c6c49a6fd84f257f7dce452e8d70e41dcbe152",
    "ultralytics_version": "8.4.145",
    "score_threshold": 0.35,
    "instance_id": "traffic-001",
    "class_id": 5,
    "class_name": "bus",
    "score": 0.925745,
    "coordinate_unit": "pixel",
    "bbox_format": "xyxy",
    "bbox_xyxy": [
      95.3,
      188.72,
      224.08,
      319.96
    ],
    "polygon_point_count": 120,
    
  },
- Polygon bổ sung chi tiết gì so với box?
Box chỉ biểu diễn object bằng hình chữ nhật xyxy, nên có thể chứa cả phần nền xung quanh. Polygon/mask mô tả sát đường viền thực tế của từng object, giúp biết chính xác vùng pixel thuộc về object, kể cả hình dạng không vuông hoặc bị cong. Trong ví dụ, polygon của chiếc bus có polygon_point_count: 120, còn box chỉ là vùng bao ngoài
- `instance_id` dùng để làm gì và không phải loại ID nào?
instance_id dùng để phân biệt từng object riêng biệt trong cùng một ảnh, ví dụ traffic-001, traffic-002. Nó giúp liên kết class, score, box và polygon của cùng một object. Đây không phải class_id, không phải ID của lớp bus, cũng không phải coco_image_id hay ID ground truth.
- Đề xuất một quy tắc biên mask:
Mask phải bám sát đường viền phần object nhìn thấy, bao phủ đầy đủ object nhưng không lấy thêm nền. Không gộp các instance riêng biệt vào cùng một mask. Với chi tiết quá nhỏ hoặc không thể xác định rõ, annotator cần theo quy tắc phân giải tối thiểu của guideline và không tự suy đoán.
- Với vùng mờ/tiếp xúc/che khuất, điều gì cần guideline hoặc escalation quyết định?
Guideline cần quy định cách vẽ mask khi object bị mờ, bị che, chạm object khác hoặc bị cắt ở mép ảnh: chỉ vẽ phần nhìn thấy hay ước lượng phần bị khuất, và mức độ nhìn thấy tối thiểu để chấp nhận. Nếu ranh giới không rõ, hai object dính vào nhau hoặc không chắc mask thuộc instance nào, annotator nên đánh dấu ambiguous và chuyển reviewer quyết định.
## 4. Vòng đời và kiểm tra chất lượng

`ảnh thô → guideline → ground truth → huấn luyện → prediction → QC/rework`

| Tác vụ | Đơn vị/định dạng ground truth | Lỗi hoặc điểm mơ hồ quan sát được | Annotator làm gì? | Reviewer xem gì? |
| --- | --- | --- | --- | --- |
| Phân loại ảnh | Một nhãn cho toàn ảnh: `class_id`, `class_name`, `taxonomy_name` | Ảnh có nhiều chủ thể, chủ thể bị che khuất, ảnh mờ hoặc không thuộc taxonomy | Xác định nhãn theo guideline; chọn `unknown` hoặc `ambiguous` nếu không chắc chắn | Kiểm tra nhãn có phù hợp với toàn ảnh và taxonomy hay không |
| Phát hiện vật thể | Một record cho mỗi object: `class_id`, `class_name`, `bbox_xyxy` theo pixel | Bỏ sót object, box chứa nhiều nền, box gộp nhiều object, object bị che hoặc cắt mép | Vẽ box sát phần nhìn thấy của đúng object và gán class; chuyển escalation khi không rõ | Kiểm tra class, số lượng object, vị trí và độ chặt của từng box |
| Instance segmentation | Một record cho mỗi instance: `instance_id`, `class_id`, `class_name`, polygon/mask | Mask vượt ra ngoài object, thiếu vùng object, hai instance bị dính hoặc ranh giới không rõ | Vẽ polygon theo biên object; giữ riêng từng instance; đánh dấu `ambiguous` khi cần | Kiểm tra biên mask, sự đầy đủ của vùng object và việc phân biệt các instance |

## 5. An toàn dữ liệu

- Một quy tắc bảo vệ dữ liệu:
Chỉ sử dụng ảnh và dữ liệu trong đúng phạm vi của bài thực hành; không chia sẻ, sao chép hoặc đưa dữ liệu lên nền tảng bên ngoài. Không ghi thông tin cá nhân vào báo cáo hoặc file output.
- Nếu thấy ảnh hoặc dữ liệu không đúng phạm vi, tôi sẽ dừng và báo cho:
Tôi sẽ dừng việc xử lý, không tải xuống hoặc chia sẻ dữ liệu đó, giữ nguyên bằng chứng cần thiết và báo ngay cho giảng viên hoặc người phụ trách bài thực hành để được hướng dẫn.
## 6. Danh sách bằng chứng

- [x ] `classification_predictions.json`
- [ x] `detection_predictions.json`
- [ x] `segmentation_predictions.json`
- [ ] `IMAGE_ATTRIBUTION.md`
- [ ] `visuals/classification_top5.png`
- [ x] `visuals/detection_predictions.png`
- [ x] `visuals/segmentation_prediction.png`
- [ ] Ô validation cuối notebook báo `PASS`.
- [ ] Không có họ tên, MSSV hoặc dữ liệu nhạy cảm trong báo cáo/output.

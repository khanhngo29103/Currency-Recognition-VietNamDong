import tkinter as tk
from tkinter import filedialog, ttk
from PIL import Image, ImageTk
import cv2
import torch
from ultralytics import YOLO
import datetime

# Tải mô hình YOLOv8
model = YOLO('kaggle/working/runs/detect/train2/weights/best.pt')

# Biến trạng thái để kiểm soát việc cập nhật khung hình từ camera
running = False
cap = None  # Để giữ tham chiếu tới đối tượng VideoCapture
countdown = 3  # Thời gian đếm ngược
image_path = ""
trigger_recognition = False  # Biến để kiểm soát việc nhận dạng từ camera
confidence_threshold = 0.5  # Ngưỡng độ tin cậy

# Hàm để nhận dạng tiền tệ trong ảnh
def recognize_currency(image):
    results = model(image)
    return results

# Hàm để mở ảnh
def open_image():
    global image_path
    image_path = filedialog.askopenfilename()
    if image_path:
        path_entry.config(state='normal')
        path_entry.delete(0, tk.END)
        path_entry.insert(0, image_path)
        path_entry.config(state='readonly')
        image = cv2.imread(image_path)
        show_image(image, img_label_left)
        update_status("Đã tải ảnh thành công")

# Hàm để hiển thị ảnh gốc
def show_image(image, label):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(image_rgb)
    pil_image.thumbnail((label.winfo_width(), label.winfo_height()))
    tk_image = ImageTk.PhotoImage(pil_image)

    label.config(image=tk_image)
    label.image = tk_image

# Hàm để hiển thị kết quả nhận dạng
def result_image():
    if image_path:
        image = cv2.imread(image_path)
        results = recognize_currency(image)
        denominations = extract_denominations(results)
        show_image_with_results(image, results)
        update_status(f"Nhận dạng hoàn tất: {denominations}")

# Hàm để trích xuất mệnh giá từ kết quả nhận dạng
def extract_denominations(results):
    denominations = []
    for result in results:
        boxes = result.boxes
        for box in boxes:
            cls = int(box.cls[0])
            confidence = box.conf[0]
            if confidence >= confidence_threshold:
                label = model.names[cls]
                denominations.append(label)
    return ', '.join(denominations)

# Hàm để vẽ kết quả lên ảnh với màu sắc và kích thước theo yêu cầu
def draw_results(image, results):
    for result in results:
        boxes = result.boxes
        for box in boxes:
            confidence = box.conf[0]
            if confidence >= confidence_threshold:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls = int(box.cls[0])
                label = f'{model.names[cls]} {confidence:.2f}'

                # Đặt màu và kích thước viền, kích cỡ chữ
                color = (0, 255, 0)  # Mặc định là màu xanh lá
                if model.names[cls] in ['10000', '20000', '500000', '100000']:
                    color = (0, 0, 255)  # Màu đỏ
                elif model.names[cls] in ['50000', '200000']:
                    color = (0, 255, 0)  # Màu xanh lá

                cv2.rectangle(image, (x1, y1), (x2, y2), color, 6)  # Tăng kích thước viền lên 6

                text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 2, 6)[0]  # Tăng kích cỡ chữ và độ dày
                text_x = x1 + (x2 - x1 - text_size[0]) // 2
                text_y = y1 + (y2 - y1 + text_size[1]) // 2
                cv2.putText(image, label, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 2, color, 6)
    return image

# Hàm để hiển thị ảnh với kết quả
def show_image_with_results(image, results):
    image_with_results = draw_results(image.copy(), results)
    show_image(image_with_results, img_label_right)

# Hàm để nhận dạng tiền từ camera và cập nhật khung hình liên tục
def recognize_from_camera():
    global running, cap, countdown, trigger_recognition
    if not running:
        running = True
        countdown = 3
        update_status("Camera sẽ bắt đầu sau 3 giây")

        def start_camera():
            global cap, running, countdown
            cap = cv2.VideoCapture(camera_combo.current())
            update_status("Camera đã bắt đầu")

            def update_frame():
                if not running:
                    cap.release()
                    update_status("Camera đã dừng")
                    return

                ret, frame = cap.read()
                if not ret:
                    return

                # Lật frame nếu bị ngược hoặc mirrored
                frame = cv2.flip(frame, 1)  # Flip horizontally
                # frame = cv2.rotate(frame, cv2.ROTATE_180)  # Rotate if upside down

                if trigger_recognition:
                    results = recognize_currency(frame)
                    frame_with_results = draw_results(frame, results)
                    show_image(frame_with_results, img_label_left)
                else:
                    show_image(frame, img_label_left)

                img_label_left.after(10, update_frame)

            update_frame()

        def countdown_timer():
            global countdown
            if countdown > 0:
                countdown_label.config(text=f"Thời gian còn lại: {countdown}s", font=('Helvetica', 14, 'bold'))  # Tăng cỡ chữ
                countdown -= 1
                countdown_label.after(1000, countdown_timer)
            else:
                countdown_label.config(text="")
                start_camera()

        countdown_timer()

# Hàm để kích hoạt nhận dạng từ camera
def trigger_recognition_toggle():
    global trigger_recognition
    trigger_recognition = not trigger_recognition
    if trigger_recognition:
        update_status("Nhận dạng đang bật")
    else:
        update_status("Nhận dạng đã tắt")

# Hàm để dừng việc nhận dạng từ camera
def stop_recognition():
    global running, cap
    running = False
    if cap:
        cap.release()
    clear_frame(img_label_left)
    update_status("Camera đã dừng thủ công")

# Hàm để làm mới ứng dụng
def refresh_app():
    global running, cap, image_path
    running = False
    image_path = ""
    if cap:
        cap.release()
    clear_frame(img_label_left)
    clear_frame(img_label_right)
    img_label_left.config(image='')
    img_label_right.config(image='')
    path_entry.config(state='normal')
    path_entry.delete(0, tk.END)
    path_entry.config(state='readonly')
    update_status("Đã làm mới ứng dụng")

# Hàm để cập nhật trạng thái
def update_status(message):
    status_label.config(text=f"TRẠNG THÁI: {message}", font=('Helvetica', 14, 'bold'), bg='blue', fg='white')  # Tăng cỡ chữ và màu nền

# Hàm để liệt kê các thiết bị camera có sẵn
def list_cameras():
    index = 0
    arr = []
    while True:
        cap = cv2.VideoCapture(index)
        if not cap.read()[0]:
            break
        else:
            arr.append(f'Camera {index}')
        cap.release()
        index += 1
    return arr

# Hàm để xác nhận lựa chọn mode
def confirm_mode():
    stop_recognition()  # Dừng camera nếu đang chạy
    refresh_app()  # Làm mới ứng dụng
    selected_option = combo.get()
    if selected_option == "Nhận dạng bằng hình ảnh":
        video_frame.pack_forget()
        picture_frame.pack(side=tk.TOP, pady=10)
        image_frame.pack(side=tk.TOP, pady=10, fill="both", expand=True)
    elif selected_option == "Nhận dạng bằng camera":
        picture_frame.pack_forget()
        image_frame.pack(side=tk.TOP, pady=10, fill="both", expand=True)
        video_frame.pack(side=tk.TOP, pady=10)

# Hàm để xóa nội dung của frame
def clear_frame(frame):
    for widget in frame.winfo_children():
        widget.destroy()

# Hàm để bắt đầu nhận dạng dựa trên lựa chọn của combobox
def start_recognition():
    selected_option = combo.get()
    if selected_option == "Nhận dạng bằng hình ảnh":
        open_image()
    elif selected_option == "Nhận dạng bằng camera":
        recognize_from_camera()

# Hàm để cập nhật ngưỡng độ tin cậy
def update_confidence_threshold():
    global confidence_threshold
    confidence_threshold = confidence_value.get()
    update_status(f"Đã cập nhật ngưỡng độ tin cậy: {confidence_threshold:.2f}")

# Tạo cửa sổ chính
window = tk.Tk()
window.title("Nhận dạng tiền tệ Việt Nam")
window.state('zoomed')  # Mở toàn màn hình ngay từ đầu

# Khung chọn chế độ
mode_frame = tk.Frame(window)
mode_frame.pack(side=tk.TOP, pady=10)

mode_label = tk.Label(mode_frame, text="Chọn chế độ nhận dạng:", font=('Helvetica', 14, 'bold'))  # Tăng cỡ chữ
mode_label.pack(side=tk.LEFT, padx=10)

combo = ttk.Combobox(mode_frame, values=["Nhận dạng bằng hình ảnh", "Nhận dạng bằng camera"], font=('Helvetica', 12))  # Tăng cỡ chữ
combo.current(0)
combo.pack(side=tk.LEFT, padx=10)

confirm_button = tk.Button(mode_frame, text="Xác nhận", command=confirm_mode, font=('Helvetica', 12, 'bold'))  # Tăng cỡ chữ
confirm_button.pack(side=tk.LEFT, padx=10)

# Khung chọn ảnh (ban đầu không hiển thị)
picture_frame = tk.Frame(window)

path_label = tk.Label(picture_frame, text="Đường dẫn ảnh:", font=('Helvetica', 14, 'bold'))  # Tăng cỡ chữ
path_label.pack(side=tk.LEFT, padx=10)

path_entry = tk.Entry(picture_frame, width=50, font=('Helvetica', 12))  # Tăng cỡ chữ
path_entry.pack(side=tk.LEFT, padx=10)
path_entry.config(state='readonly')

browse_button = tk.Button(picture_frame, text="Chọn ảnh", command=open_image, font=('Helvetica', 12, 'bold'))  # Tăng cỡ chữ
browse_button.pack(side=tk.LEFT, padx=10)

result_button = tk.Button(picture_frame, text="Nhận dạng", command=result_image, font=('Helvetica', 12, 'bold'))  # Tăng cỡ chữ
result_button.pack(side=tk.LEFT, padx=10)

# Khung hiển thị ảnh (ban đầu không hiển thị)
image_frame = tk.Frame(window)

image_frame_left = tk.Frame(image_frame, bd=2, relief=tk.SUNKEN)
image_frame_left.pack(side=tk.LEFT, padx=10, pady=10, fill="both", expand=True)

img_label_left = tk.Label(image_frame_left)
img_label_left.pack(fill="both", expand=True)

image_frame_right = tk.Frame(image_frame, bd=2, relief=tk.SUNKEN)
image_frame_right.pack(side=tk.RIGHT, padx=10, pady=10, fill="both", expand=True)

img_label_right = tk.Label(image_frame_right)
img_label_right.pack(fill="both", expand=True)

# Khung hiển thị video (ban đầu không hiển thị)
video_frame = tk.Frame(window)

camera_label = tk.Label(video_frame, text="Chọn camera:", font=('Helvetica', 14, 'bold'))  # Tăng cỡ chữ
camera_label.pack(side=tk.LEFT, padx=10)

camera_combo = ttk.Combobox(video_frame, values=list_cameras(), font=('Helvetica', 12))  # Tăng cỡ chữ
camera_combo.current(0)
camera_combo.pack(side=tk.LEFT, padx=10)

camera_button = tk.Button(video_frame, text="Bắt đầu camera", command=recognize_from_camera, font=('Helvetica', 12, 'bold'))  # Tăng cỡ chữ
camera_button.pack(side=tk.LEFT, padx=10)

trigger_button = tk.Button(video_frame, text="Bật/Tắt nhận dạng", command=trigger_recognition_toggle, font=('Helvetica', 12, 'bold'))  # Tăng cỡ chữ
trigger_button.pack(side=tk.LEFT, padx=10)

stop_button = tk.Button(video_frame, text="Dừng camera", command=stop_recognition, font=('Helvetica', 12, 'bold'))  # Tăng cỡ chữ
stop_button.pack(side=tk.LEFT, padx=10)

countdown_label = tk.Label(video_frame, text="", font=('Helvetica', 14, 'bold'))  # Tăng cỡ chữ
countdown_label.pack(side=tk.LEFT, padx=10)

# Khung trạng thái
status_frame = tk.Frame(window, bg="blue")
status_frame.pack(fill="x")

status_label = tk.Label(status_frame, text="TRẠNG THÁI: OK", font=('Helvetica', 12, 'bold'), bg="blue", fg="white")
status_label.pack(side=tk.LEFT, padx=10)

# Frame cho hiển thị thời gian hiện tại
def update_time():
    current_time = datetime.datetime.now().strftime("%d-%m-%Y %H:%M:%S")
    time_label.config(text=current_time)
    window.after(1000, update_time)

time_label = tk.Label(status_frame, font=('calibri', 12, 'bold'), background='blue', foreground='white')
time_label.pack(side=tk.RIGHT, padx=10)
update_time()

# Khung điều chỉnh ngưỡng độ tin cậy
confidence_frame = tk.Frame(window)
confidence_frame.pack(side=tk.BOTTOM, pady=10)

confidence_label = tk.Label(confidence_frame, text="Ngưỡng độ tin cậy:", font=('Helvetica', 14, 'bold'))  # Tăng cỡ chữ
confidence_label.pack(side=tk.LEFT, padx=10)

confidence_value = tk.DoubleVar(value=confidence_threshold)
confidence_scale = tk.Scale(confidence_frame, from_=0.0, to=1.0, resolution=0.01, orient=tk.HORIZONTAL, variable=confidence_value, command=lambda val: update_confidence_threshold(), font=('Helvetica', 12))  # Tăng cỡ chữ
confidence_scale.pack(side=tk.LEFT, padx=10)

refresh_button = tk.Button(window, text="Làm mới ứng dụng", command=refresh_app, font=('Helvetica', 12, 'bold'))  # Tăng cỡ chữ
refresh_button.pack(side=tk.BOTTOM, pady=10)

window.mainloop()

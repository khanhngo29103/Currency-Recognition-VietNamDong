import tkinter as tk
from tkinter import filedialog, ttk
from PIL import Image, ImageTk
import cv2
import torch
from ultralytics import YOLO
import datetime

# Tải mô hình YOLO
model = YOLO('best.pt')  # Thay 'yolov8s.pt' bằng đường dẫn tới mô hình của bạn

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

# Hàm để vẽ kết quả lên ảnh
def draw_results(image, results):
    for result in results:
        boxes = result.boxes
        for box in boxes:
            confidence = box.conf[0]
            if confidence >= confidence_threshold:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls = int(box.cls[0])
                label = f'{model.names[cls]} {confidence:.2f}'
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                text_x = x1
                text_y = y1 - 10 if y1 - 10 > 10 else y1 + 10
                cv2.putText(image, label, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
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
                countdown_label.config(text=f"Thời gian còn lại: {countdown}s")
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
    status_label.config(text=f"TRẠNG THÁI: {message}")

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
        image_frame_left.pack(side=tk.LEFT, padx=10, pady=10, fill="both", expand=True)
        image_frame_right.pack(side=tk.RIGHT, padx=10, pady=10, fill="both", expand=True)
    elif selected_option == "Nhận dạng bằng camera":
        picture_frame.pack_forget()
        image_frame_left.pack(side=tk.TOP, pady=10, fill="both", expand=True)
        image_frame_right.pack_forget()
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
    update_status(f"Đã cập nhật ngưỡng độ tin cậy: {confidence_threshold:.1f}")

# Tạo giao diện Tkinter
root = tk.Tk()
root.title("CHƯƠNG TRÌNH NHẬN DẠNG TIỀN")

# Thiết lập kích thước và vị trí cửa sổ để phóng to
root.state('zoomed')

# Frame cho trạng thái
status_frame = tk.Frame(root, bg="blue")
status_frame.pack(fill="x")

status_label = tk.Label(status_frame, text="TRẠNG THÁI: OK", font=('Helvetica', 12, 'bold'), bg="blue", fg="white")
status_label.pack(side=tk.LEFT, padx=10)

# Frame cho hiển thị thời gian hiện tại
def update_time():
    current_time = datetime.datetime.now().strftime("%d-%m-%Y %H:%M:%S")
    time_label.config(text=current_time)
    root.after(1000, update_time)

time_label = tk.Label(status_frame, font=('calibri', 12, 'bold'), background='blue', foreground='white')
time_label.pack(side=tk.RIGHT, padx=10)
update_time()

# Frame chính
main_frame = tk.Frame(root)
main_frame.pack(fill="both", expand=True)

# Tạo hai frame cho ảnh gốc và ảnh nhận dạng
image_frame_left = tk.Frame(main_frame, bd=2, relief="sunken", width=400, height=600)
image_frame_right = tk.Frame(main_frame, bd=2, relief="sunken", width=400, height=600)

img_label_left = tk.Label(image_frame_left)
img_label_left.pack(fill="both", expand=True)
img_label_right = tk.Label(image_frame_right)
img_label_right.pack(fill="both", expand=True)

# Nhãn để hiển thị bộ đếm thời gian
countdown_label = tk.Label(main_frame, font=('Helvetica', 12), bg="white", fg="black")
countdown_label.pack()

# Frame cho các cài đặt và điều khiển
control_frame = tk.Frame(main_frame, width=300)  # Tăng độ rộng của control_frame
control_frame.pack(side=tk.RIGHT, fill="y")

# Combobox để chọn chế độ nhận dạng
mode_label = tk.Label(control_frame, text="Mode :", font=('Helvetica', 12, 'bold'))
mode_label.pack(pady=5)
combo_options = ["Nhận dạng bằng hình ảnh", "Nhận dạng bằng camera"]
combo = ttk.Combobox(control_frame, values=combo_options, width=25, font=('Helvetica', 12))
combo.set(combo_options[0])
combo.pack()
confirm_button = tk.Button(control_frame, text="XÁC NHẬN", command=confirm_mode, width=20, bg="red", fg="white", font=('Helvetica', 12))
confirm_button.pack(pady=5)

# Frame cho cài đặt camera
video_frame = tk.Frame(control_frame)
camera_label = tk.Label(video_frame, text="Camera :", font=('Helvetica', 12))
camera_label.pack()
camera_options = list_cameras()
camera_combo = ttk.Combobox(video_frame, values=camera_options, width=25, font=('Helvetica', 12))
if camera_options:
    camera_combo.set(camera_options[0])
camera_combo.pack()
start_button = tk.Button(video_frame, text="BẮT ĐẦU", width=20, command=start_recognition, bg="green", fg="white", font=('Helvetica', 12))
start_button.pack(pady=5)
stop_button = tk.Button(video_frame, text="DỪNG", width=20, command=stop_recognition, bg="red", fg="white", font=('Helvetica', 12))
stop_button.pack(pady=5)
trigger_button = tk.Button(video_frame, text="KÍCH HOẠT NHẬN DẠNG", width=20, command=trigger_recognition_toggle, bg="yellow", fg="black", font=('Helvetica', 12))
trigger_button.pack(pady=5)

# Frame cho chọn ảnh
picture_frame = tk.Frame(control_frame)
path_label = tk.Label(picture_frame, text="Đường dẫn :", font=('Helvetica', 12))
path_label.pack()
path_entry = tk.Entry(picture_frame, width=30, state='readonly', font=('Helvetica', 12))
path_entry.pack()
choose_button = tk.Button(picture_frame, text="CHỌN", width=20, command=open_image, bg="red", fg="white", font=('Helvetica', 12))
choose_button.pack(pady=5)
result_button = tk.Button(picture_frame, text="KẾT QUẢ", width=20, command=result_image, bg="green", fg="white", font=('Helvetica', 12))
result_button.pack(pady=5)

# Các cài đặt
setting_label = tk.Label(control_frame, text="CÀI ĐẶT", font=('Helvetica', 12, 'bold'))
setting_label.pack(pady=5)

confidence_label = tk.Label(control_frame, text="Độ tin cậy :", font=('Helvetica', 12))
confidence_label.pack()
confidence_value = tk.DoubleVar(value=0.5)
confidence_spinbox = ttk.Spinbox(control_frame, from_=0, to=1, increment=0.1, textvariable=confidence_value, width=10, font=('Helvetica', 12))
confidence_spinbox.pack()

save_button = tk.Button(control_frame, text="LƯU", width=20, command=update_confidence_threshold, font=('Helvetica', 12))
save_button.pack(pady=10)

refresh_button = tk.Button(control_frame, text="LÀM MỚI", width=20, command=refresh_app, bg="blue", fg="white", font=('Helvetica', 12))
refresh_button.pack(pady=10)

exit_button = tk.Button(control_frame, text="THOÁT", width=20, command=root.quit, bg="red", fg="white", font=('Helvetica', 12))
exit_button.pack(pady=10)

root.mainloop()

import flet as ft
import cv2
import base64
import os
import sys
import threading
import traceback
import subprocess
import flet.canvas as cv
from main import TennisMatchProcessor

def main(page: ft.Page):
    page.title = "Tenis Analiz Sistemi"
    page.theme_mode = ft.ThemeMode.DARK
    page.padding = 30
    page.scroll = ft.ScrollMode.AUTO 

    # --- Global Variables ---
    input_video_path = None
    corners = []
    processor = TennisMatchProcessor()

    # --- UI Components ---
    
    # 1. Upload Section Components
    path_field = ft.TextField(
        label="Dosya Yolunu Yapıştır", 
        hint_text="/Users/Name/video.mp4", 
        text_size=14, 
        expand=True,
        border_color=ft.Colors.BLUE_200
    )
    
    loading_info = ft.Text("", color=ft.Colors.YELLOW)

    def on_file_picked(e: ft.FilePickerResultEvent):
        if e.files:
            handle_video_input(e.files[0].path)

    file_picker = ft.FilePicker(on_result=on_file_picked)
    page.overlay.append(file_picker)

    def on_manual_submit(e):
        handle_video_input(path_field.value)

    def on_drop(e):
        handle_video_input(e.data)

    page.on_file_drop = on_drop

    drag_area = ft.Container(
        content=ft.Column(
            [
                ft.Icon(ft.Icons.CLOUD_UPLOAD, size=80, color=ft.Colors.WHITE),
                ft.Text("Video Dosyasını Sürükleyip Buraya Bırakın", size=20, weight=ft.FontWeight.BOLD),
            ],
            alignment=ft.MainAxisAlignment.CENTER,
            horizontal_alignment=ft.CrossAxisAlignment.CENTER
        ),
        bgcolor=ft.Colors.LIGHT_BLUE_500,
        border_radius=15,
        alignment=ft.alignment.center,
        height=400, # Large drop area
        ink=True,
        on_click=lambda _: file_picker.pick_files(allow_multiple=False, allowed_extensions=["mp4", "avi", "mov"]),
    )

    def open_file_picker(e):
        print("Dosya seçme butonu tıklandı.")
        
        if sys.platform == "darwin":
            try:
                # Use osascript to open native macOS file dialog
                cmd = """osascript -e 'get POSIX path of (choose file of type {"mp4"} with prompt "Analiz edilecek videoyu seçin")'"""
                result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
                
                if result.returncode == 0:
                    selected_path = result.stdout.strip()
                    if selected_path:
                        print(f"Seçilen dosya (macOS): {selected_path}")
                        path_field.value = selected_path
                        path_field.update()
                        handle_video_input(selected_path)
                else:
                    print("Dosya seçimi iptal edildi veya hata oluştu.")
            except Exception as ex:
                print(f"macOS file picker hatası: {ex}")
                # Fallback to Flet picker if osascript fails
                file_picker.pick_files(allow_multiple=False, allowed_extensions=["mp4"])
        else:
            try:
                file_picker.pick_files(allow_multiple=False, allowed_extensions=["mp4"])
                print("File picker dialog açılmalı.")
            except Exception as ex:
                print(f"File picker hatası: {ex}")

    upload_view = ft.Column(
        [
            ft.Text("Analiz Edilecek Maç Videosunu Buraya Yükleyin", size=28, weight=ft.FontWeight.BOLD),
            drag_area,
            ft.Row(
                [
                    path_field,
                    ft.ElevatedButton(
                        "Dosyayı Seç", 
                        icon=ft.Icons.FOLDER_OPEN, 
                        on_click=open_file_picker,
                        bgcolor=ft.Colors.LIGHT_BLUE_600,
                        color=ft.Colors.WHITE
                    ),
                    ft.ElevatedButton(
                        "Yükle", 
                        on_click=on_manual_submit,
                        bgcolor=ft.Colors.GREEN_600,
                        color=ft.Colors.WHITE
                    )
                ],
                alignment=ft.MainAxisAlignment.SPACE_BETWEEN
            ),
            loading_info
        ],
        spacing=30,
        alignment=ft.MainAxisAlignment.CENTER
    )

    # 2. Corner Selection Components
    frame_image = ft.Image(src_base64="", width=640, height=360, fit=ft.ImageFit.CONTAIN)
    
    canvas = cv.Canvas(
        shapes=[],
        width=640,
        height=360,
    )

    instruction_text = ft.Text("Kortun 4 köşesine sırasıyla tıklayın (Sol-Üst, Sağ-Üst, Sağ-Alt, Sol-Alt)", size=16)
    
    def reset_corners(e=None):
        corners.clear()
        canvas.shapes.clear()
        canvas.update()
        start_btn.disabled = True
        instruction_text.value = "Kortun 4 köşesine sırasıyla tıklayın..."
        page.update()

    start_btn = ft.ElevatedButton("Analizi Başlat", disabled=True, on_click=lambda _: start_analysis_thread(), bgcolor=ft.Colors.GREEN_700, color=ft.Colors.WHITE)
    reset_btn = ft.ElevatedButton("Sıfırla", on_click=reset_corners, bgcolor=ft.Colors.RED_700, color=ft.Colors.WHITE)

    def on_canvas_tap(e: ft.TapEvent):
        if len(corners) >= 4: return
        x, y = e.local_x, e.local_y
        corners.append((x, y))
        
        # Draw dot
        canvas.shapes.append(cv.Circle(x, y, 5, ft.Paint(color=ft.Colors.RED, style=ft.PaintingStyle.FILL)))
        
        # Draw lines
        if len(corners) > 1:
            p1 = corners[-2]
            p2 = corners[-1]
            canvas.shapes.append(cv.Line(p1[0], p1[1], p2[0], p2[1], ft.Paint(color=ft.Colors.GREEN, stroke_width=2)))
        
        if len(corners) == 4:
            p1 = corners[-1]
            p2 = corners[0]
            canvas.shapes.append(cv.Line(p1[0], p1[1], p2[0], p2[1], ft.Paint(color=ft.Colors.GREEN, stroke_width=2)))
            start_btn.disabled = False
            instruction_text.value = "Köşeler tamamlandı! Analizi başlatabilirsiniz."
        
        canvas.update()
        start_btn.update()
        instruction_text.update()

    selection_stack = ft.Stack(
        [
            frame_image,
            canvas,
            ft.GestureDetector(
                on_tap_down=on_canvas_tap,
                content=ft.Container(bgcolor=ft.Colors.TRANSPARENT, width=640, height=360)
            )
        ],
        width=640, height=360
    )

    selection_view = ft.Column(
        [
            ft.Text("Kortun Köşelerini Seç", size=28, weight=ft.FontWeight.BOLD),
            ft.Container(selection_stack, border=ft.border.all(1, ft.Colors.GREY_700), border_radius=8),
            instruction_text,
            ft.Row([reset_btn, start_btn], alignment=ft.MainAxisAlignment.CENTER, spacing=20)
        ],
        visible=False,
        horizontal_alignment=ft.CrossAxisAlignment.CENTER,
        spacing=20
    )

    # 3. Processing View
    progress_bar = ft.ProgressBar(width=400, color=ft.Colors.BLUE)
    status_text = ft.Text("İşleniyor...", size=16)
    processing_view = ft.Column(
        [
            ft.Text("Analiz Yapılıyor...", size=24, weight=ft.FontWeight.BOLD),
            progress_bar,
            status_text
        ],
        visible=False,
        horizontal_alignment=ft.CrossAxisAlignment.CENTER,
        alignment=ft.MainAxisAlignment.CENTER,
        height=500
    )


    # 4. Results View
    results_view = ft.Column(visible=False, spacing=30)
    
    # Headers
    headers = ft.Row([
        ft.Container(content=ft.Text("Maç Hakkında bilgiler", color=ft.Colors.BLACK, weight=ft.FontWeight.BOLD), bgcolor=ft.Colors.TEAL_200, padding=10, border_radius=5),
        ft.Container(content=ft.Text("Maç Analizi", color=ft.Colors.WHITE, weight=ft.FontWeight.BOLD), bgcolor=ft.Colors.BLUE_600, padding=10, border_radius=5),
    ])

    # --- Logic ---

    def handle_video_input(path):
        nonlocal input_video_path
        if not path: return
        
        # Clean path quotes if any
        path = path.strip().strip("'").strip('"')
        
        if not os.path.exists(path):
            loading_info.value = "Hata: Dosya bulunamadı."
            loading_info.update()
            return

        input_video_path = os.path.abspath(path)
        loading_info.value = f"Yükleniyor: {os.path.basename(input_video_path)}"
        loading_info.update()

        try:
            cap = cv2.VideoCapture(input_video_path)
            ret, frame = cap.read()
            cap.release()
            
            if ret:
                # Resize specifically for the Canvas UI ID
                frame_resized = cv2.resize(frame, (640, 360))
                
                _, buffer = cv2.imencode('.jpg', frame_resized)
                b64_img = base64.b64encode(buffer).decode('utf-8')
                frame_image.src_base64 = b64_img
                
                # Transition
                upload_view.visible = False
                selection_view.visible = True
                loading_info.value = ""
                page.update()
            else:
                loading_info.value = "Hata: Video karesi okunamadı."
                loading_info.update()
        except Exception as e:
            loading_info.value = f"Hata: {str(e)}"
            loading_info.update()
            print(traceback.format_exc())

    def start_analysis_thread():
        selection_view.visible = False
        processing_view.visible = True
        page.update()
        
        t = threading.Thread(target=run_analysis, daemon=True)
        t.start()

    def update_status(msg):
        status_text.value = msg
        status_text.update()

    def run_analysis():
        try:
            # Scale corners back to original video? 
            # Original video resolution might not be 640x360.
            # We need to know original size to scale corners.
            
            cap = cv2.VideoCapture(input_video_path)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.release()
            
            # Canvas is 640x360.
            scale_x = width / 640
            scale_y = height / 360
            
            real_corners = []
            for (cx, cy) in corners:
                real_corners.append(int(cx * scale_x))
                real_corners.append(int(cy * scale_y)) 
                # Note: utils expects flat list? No, ActionFilter and others expect specific format.
                # select_corners_manually returns list of points.
                # Let's check main.py expected format. It expects whatever select_corners_manually returns.
                # select_corners returns list of [x, y].
            
            # Format corners as [x,y, x,y...] or list of tuples? 
            # Let's re-read main.py logic if needed. 
            # select_corners_manually returns simple list of coords.
            # Actually Main code expects a list of points (x, y). 
            # But flattening might be needed if downstream uses it that way. 
            # Looking at previous app.py, it passed `corners` list directly. 
            # Corner list was list of tuples or lists. 
            
            # Let's pass simple flat list if needed? 
            # No, main.py -> process_match -> ActionFilter uses `roi_corners`.
            # ActionFilter expects 4 points.
            
            out_vid, out_mini, stats_df = processor.process_match(
                input_video_path, 
                corners=real_corners, 
                progress_callback=update_status
            )
            
            if out_vid is None:
                update_status("Analiz başarısız: Video oluşturulamadı veya aksiyon tespit edilemedi.")
                return

            show_results(out_vid, out_mini, stats_df)
            
        except Exception as e:
            err = traceback.format_exc()
            print(err)
            update_status(f"Hata oluştu: {str(e)}")

    def open_externally(path):
        if not path or not os.path.exists(path): return
        try:
            if os.name == 'nt': # Windows
                os.startfile(path)
            elif os.name == 'posix': # Mac/Linux
                subprocess.run(['open', path])
        except Exception as e:
            print(f"Error opening external: {e}")

    def show_results(out_vid, out_mini, stats_df):
        processing_view.visible = False
        
        # Build Results Grid
        # Left: Main Video (Large)
        # Right: Mini Court (Small)
        
        main_video = ft.Video(
            playlist=[ft.VideoMedia(out_vid)],
            playlist_mode=ft.PlaylistMode.LOOP,
            aspect_ratio=16/9,
            autoplay=True,
            filter_quality=ft.FilterQuality.HIGH,
            muted=True # Mute main to avoid noise
        )
        
        mini_video = ft.Video(
            playlist=[ft.VideoMedia(out_mini)],
            playlist_mode=ft.PlaylistMode.LOOP,
            aspect_ratio=350/600,
            autoplay=True,
            filter_quality=ft.FilterQuality.HIGH,
            muted=True
        )
        
        # External Open Buttons
        btn_open_main = ft.ElevatedButton(
            "PC'de Aç (Ana Video)", 
            icon=ft.Icons.OPEN_IN_NEW,
            on_click=lambda _: open_externally(out_vid)
        )
        
        btn_open_mini = ft.ElevatedButton(
            "PC'de Aç (Mini Kort)",
            icon=ft.Icons.OPEN_IN_NEW,
            on_click=lambda _: open_externally(out_mini)
        )
        
        # Stats Table
        stats_content = ft.Container(content=ft.Text("İstatistik alınamadı."), padding=10)
        
        if stats_df is not None and not stats_df.empty:
            last_row = stats_df.iloc[-1]
            dt = ft.DataTable(
                columns=[
                    ft.DataColumn(ft.Text("İstatistik")),
                    ft.DataColumn(ft.Text("Oyuncu 1")),
                    ft.DataColumn(ft.Text("Oyuncu 2")),
                ],
                rows=[
                    ft.DataRow(cells=[
                        ft.DataCell(ft.Text("Toplam Şut")),
                        ft.DataCell(ft.Text(f"{int(last_row['player_1_number_of_shots'])}")),
                        ft.DataCell(ft.Text(f"{int(last_row['player_2_number_of_shots'])}")),
                    ]),
                    ft.DataRow(cells=[
                        ft.DataCell(ft.Text("Son Şut Hızı (km/h)")),
                        ft.DataCell(ft.Text(f"{last_row['player_1_last_shot_speed']:.2f}")),
                        ft.DataCell(ft.Text(f"{last_row['player_2_last_shot_speed']:.2f}")),
                    ]),
                     ft.DataRow(cells=[
                        ft.DataCell(ft.Text("Koşu Mesafesi (m)")),
                        ft.DataCell(ft.Text(f"{last_row['player_1_total_player_speed']:.2f}")),
                        ft.DataCell(ft.Text(f"{last_row['player_2_total_player_speed']:.2f}")),
                    ]),
                ],
                border=ft.border.all(1, ft.Colors.GREY_800),
                vertical_lines=ft.border.BorderSide(1, ft.Colors.GREY_800),
                horizontal_lines=ft.border.BorderSide(1, ft.Colors.GREY_800),
            )
            stats_content = ft.Container(
                content=ft.Column(
                    [
                        ft.Text("Oyuncu İstatistikleri", size=20, weight=ft.FontWeight.BOLD),
                        dt
                    ]
                ),
                padding=20,
                bgcolor=ft.Colors.BLUE_GREY_900,
                border_radius=10
            )

        grid = ft.Column(
            [
                ft.Row(
                    [
                        # Left Column: Main Video
                        ft.Column(
                            [
                                ft.Row([ft.Text("Ana Video", color=ft.Colors.RED_300, weight=ft.FontWeight.BOLD), btn_open_main], alignment=ft.MainAxisAlignment.SPACE_BETWEEN),
                                ft.Container(content=main_video, width=700, height=394, border_radius=10, clip_behavior=ft.ClipBehavior.HARD_EDGE)
                            ],
                            alignment=ft.MainAxisAlignment.START
                        ),
                        # Right Column: Mini Court
                        ft.Column(
                            [
                                ft.Row([ft.Text("Mini Kort", color=ft.Colors.RED_300, weight=ft.FontWeight.BOLD), btn_open_mini], alignment=ft.MainAxisAlignment.SPACE_BETWEEN),
                                ft.Container(content=mini_video, width=230, height=394, border_radius=10, clip_behavior=ft.ClipBehavior.HARD_EDGE) # Adjusted to match height
                            ],
                            alignment=ft.MainAxisAlignment.START
                        )
                    ],
                    alignment=ft.MainAxisAlignment.START,
                    vertical_alignment=ft.CrossAxisAlignment.START,
                    spacing=20
                ),
                stats_content
            ],
            spacing=30
        )

        results_view.controls.append(headers)
        results_view.controls.append(grid)
        results_view.visible = True
        page.update()


    # Add views to page
    page.add(upload_view)
    page.add(selection_view)
    page.add(processing_view)
    page.add(results_view)

if __name__ == "__main__":
    ft.app(target=main)

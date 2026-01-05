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
    page.horizontal_alignment = ft.CrossAxisAlignment.CENTER
    page.vertical_alignment = ft.MainAxisAlignment.CENTER

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

    # 2. Scoreboard Selection View
    frame_image = ft.Image(src_base64="", width=640, height=360, fit=ft.ImageFit.CONTAIN)
    
    canvas = cv.Canvas(
        shapes=[],
        width=640,
        height=360,
    )

    instruction_text = ft.Text("İsteğe Bağlı: Skor tablosunu seçmek için 'Skor Tablosu Seç' butonuna basın veya direkt başlatın.", size=16)
    
    def reset_selection(e=None):
        scoreboard_points.clear()
        nonlocal scoreboard_roi, selection_mode
        scoreboard_roi = None
        selection_mode = "NONE"
        
        canvas.shapes.clear()
        canvas.update()
        instruction_text.value = "Seçim sıfırlandı. Analizi başlatabilir veya skor tablosu seçebilirsiniz."
        page.update()

    start_btn = ft.ElevatedButton("Analizi Başlat", on_click=lambda _: start_analysis_thread(), bgcolor=ft.Colors.GREEN_700, color=ft.Colors.WHITE)
    reset_btn = ft.ElevatedButton("Seçimi Sıfırla", on_click=reset_selection, bgcolor=ft.Colors.RED_700, color=ft.Colors.WHITE)
    
    # Selection State
    selection_mode = "NONE" 
    scoreboard_points = []
    scoreboard_roi = None # (x1, y1, x2, y2)

    def on_canvas_tap(e: ft.TapEvent):
        nonlocal selection_mode
        x, y = e.local_x, e.local_y

        if selection_mode == "SCOREBOARD":
            if len(scoreboard_points) >= 2: 
                scoreboard_points.clear()
                # Clear previous rectangle logic if needed, but for now we just draw new ones
                # Ideally we clear shapes but keep the image... 
                # For simplicity, let's just clear shapes if we restart selection
                canvas.shapes = [s for s in canvas.shapes if not isinstance(s, cv.Rect) and not isinstance(s, cv.Circle)]
                
            scoreboard_points.append((x, y))
            canvas.shapes.append(cv.Circle(x, y, 5, ft.Paint(color=ft.Colors.YELLOW, style=ft.PaintingStyle.FILL)))

            if len(scoreboard_points) == 2:
                # Draw Rectangle
                p1 = scoreboard_points[0]
                p2 = scoreboard_points[1]
                x1, y1 = min(p1[0], p2[0]), min(p1[1], p2[1])
                x2, y2 = max(p1[0], p2[0]), max(p1[1], p2[1])
                
                nonlocal scoreboard_roi
                scoreboard_roi = (x1, y1, x2, y2)
                
                canvas.shapes.append(cv.Rect(x1, y1, x2-x1, y2-y1, ft.Paint(color=ft.Colors.YELLOW, stroke_width=2, style=ft.PaintingStyle.STROKE)))
                instruction_text.value = "Skor tablosu seçildi! Analizi başlatabilir veya seçimi sıfırlayabilirsiniz."
                selection_mode = "NONE" # Reset mode after selection
                
        canvas.update()
        instruction_text.update()

    def set_scoreboard_mode(e):
        nonlocal selection_mode
        selection_mode = "SCOREBOARD"
        instruction_text.value = "Skor tablosunun Sol-Üst ve Sağ-Alt köşesine tıklayın."
        scoreboard_points.clear()
        # Clear existing calc
        nonlocal scoreboard_roi
        scoreboard_roi = None
        canvas.shapes.clear() # Clear old drawings
        canvas.update()
        instruction_text.update()
        
    scoreboard_btn = ft.ElevatedButton("Skor Tablosu Seç", on_click=set_scoreboard_mode, bgcolor=ft.Colors.ORANGE_700, color=ft.Colors.WHITE)

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
            ft.Text("Hazırlık Aşaması", size=28, weight=ft.FontWeight.BOLD),
            ft.Container(selection_stack, border=ft.border.all(1, ft.Colors.GREY_700), border_radius=8),
            instruction_text,
            ft.Row([reset_btn, scoreboard_btn, start_btn], alignment=ft.MainAxisAlignment.CENTER, spacing=20)
        ],
        visible=False,
        horizontal_alignment=ft.CrossAxisAlignment.CENTER,
        spacing=20
    )

    # 3. Processing View
    progress_bar = ft.ProgressBar(width=400, color=ft.Colors.BLUE)
    status_text = ft.Text("İşleniyor...", size=16, text_align=ft.TextAlign.CENTER)
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
            # NO MANUAL CORNERS - Using Auto Detection
            
            cap = cv2.VideoCapture(input_video_path)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.release()
            
            # Canvas is 640x360.
            scale_x = width / 640
            scale_y = height / 360
            
            real_scoreboard_roi = None
            if scoreboard_roi:
                sx1, sy1, sx2, sy2 = scoreboard_roi
                real_scoreboard_roi = (
                    int(sx1 * scale_x), int(sy1 * scale_y),
                    int(sx2 * scale_x), int(sy2 * scale_y)
                )

            out_vid, out_mini, stats_df, score_events = processor.process_match(
                input_video_path, 
                corners=None, # Auto-detect
                scoreboard_roi=real_scoreboard_roi,
                progress_callback=update_status
            )
            
            if out_vid is None:
                update_status("Analiz başarısız: Video oluşturulamadı veya aksiyon tespit edilemedi.")
                return

            show_results(out_vid, out_mini, stats_df, score_events)
            
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

    # --- 4. Interactive Results View ---
    
    class InteractiveMiniCourt(ft.Stack):
        def __init__(self, bounce_data, on_point_click):
            self.bounce_data = bounce_data # List of dicts: {pos: (x,y), timestamp: t, score: s}
            self.on_point_click = on_point_click
            
            interactive_width = 350
            interactive_height = 600
             # Mini Court Dimensions from mini_court.py (approx)
            drawing_width = 250
            drawing_height = 500
            padding = 20
            start_x = (interactive_width - drawing_width) / 2
            start_y = (interactive_height - drawing_height) / 2
            end_x = start_x + drawing_width
            end_y = start_y + drawing_height
            
            # Court inner dimensions
            court_start_x = start_x + padding
            court_start_y = start_y + padding
            court_end_x = end_x - padding
            court_end_y = end_y - padding
            c_width = court_end_x - court_start_x
            c_height = court_end_y - court_start_y

            shapes = []
            
            # 1. Background
            shapes.append(cv.Rect(start_x, start_y, drawing_width, drawing_height, 
                                  paint=ft.Paint(color=ft.Colors.GREEN_900, style=ft.PaintingStyle.FILL)))
            shapes.append(cv.Rect(start_x, start_y, drawing_width, drawing_height, 
                                  paint=ft.Paint(color=ft.Colors.WHITE, style=ft.PaintingStyle.STROKE, stroke_width=2)))
            
            # 2. Court Lines (White)
            paint_white = ft.Paint(color=ft.Colors.WHITE, stroke_width=2, style=ft.PaintingStyle.STROKE)
            
            # Outer Boundary
            shapes.append(cv.Rect(court_start_x, court_start_y, c_width, c_height, paint=paint_white))
            
            # Net
            net_y = court_start_y + c_height / 2
            shapes.append(cv.Line(court_start_x, net_y, court_end_x, net_y, paint=paint_white))
            
            # Center Service Line
            center_x = court_start_x + c_width / 2
            # Service lines approx 5.5m from baseline. Total length 23.77m.
            # Ratio: 5.485 / 23.77
            ratio = 5.485 / 23.77
            service_top_y = court_start_y + c_height * ratio
            service_bottom_y = court_end_y - c_height * ratio
            
            shapes.append(cv.Line(center_x, service_top_y, center_x, service_bottom_y, paint=paint_white))
            
            # Service Lines (Horizontal)
            shapes.append(cv.Line(court_start_x, service_top_y, court_end_x, service_top_y, paint=paint_white))
            shapes.append(cv.Line(court_start_x, service_bottom_y, court_end_x, service_bottom_y, paint=paint_white))

            # Singles Sidelines
            # 1.37m / 10.97m
            # margin_ratio = 1.37 / 10.97
            margin_ratio = 0.125
            single_left_x = court_start_x + c_width * margin_ratio
            single_right_x = court_end_x - c_width * margin_ratio
            
            shapes.append(cv.Line(single_left_x, court_start_y, single_left_x, court_end_y, paint=paint_white))
            shapes.append(cv.Line(single_right_x, court_start_y, single_right_x, court_end_y, paint=paint_white))
            
            # 3. Heatmap Points (Clickable)
            stack_controls = [
                cv.Canvas(shapes, width=interactive_width, height=interactive_height)
            ]
            
            for event in self.bounce_data:
                pos = event['pos'] # (x, y) matching the video mini court dimensions (350x600)
                timestamp = event['timestamp']
                score = event.get('score', '')
                
                # Check bounds for drawing
                if not (0 <= pos[0] <= interactive_width and 0 <= pos[1] <= interactive_height):
                    continue
                
                # Determine color based on "In/Out" logic
                # Court boundaries: (court_start_x, court_start_y) to (court_end_x, court_end_y)
                
                # Check Net Zone (Invalid shots hitting the net / near net)
                # net_y is calculated above in this scope
                net_zone_buffer = 20 # pixels +/- from net line
                is_net = (net_y - net_zone_buffer <= pos[1] <= net_y + net_zone_buffer)

                is_inside = (court_start_x <= pos[0] <= court_end_x) and \
                            (court_start_y <= pos[1] <= court_end_y) and \
                            not is_net
                
                # Color request: Inside = Light Blue, Outside or Net = Red
                point_color = ft.Colors.LIGHT_BLUE_ACCENT if is_inside else ft.Colors.RED
                
                # Create a clickable dot
                dot = ft.Container(
                    width=14, height=14,
                    border_radius=7,
                    bgcolor=point_color,
                    left=pos[0]-7, 
                    top=pos[1]-7,
                    tooltip=f"Score: {score} @ {timestamp:.1f}s ({'IN' if is_inside else 'OUT'})",
                    on_click=lambda _, t=timestamp: self.on_point_click(t)
                )
                stack_controls.append(dot)
                
            super().__init__(
                controls=stack_controls, 
                width=interactive_width, 
                height=interactive_height
            )


    def show_results(out_vid, out_mini, stats_df, bounce_events=None):
        processing_view.visible = False
        
        main_video = ft.Video(
            playlist=[ft.VideoMedia(out_vid)],
            playlist_mode=ft.PlaylistMode.LOOP,
            aspect_ratio=16/9,
            autoplay=True,
            filter_quality=ft.FilterQuality.HIGH,
            muted=True 
        )
        
        # Mini Court Video (The "Old" View)
        mini_video = ft.Video(
            playlist=[ft.VideoMedia(out_mini)],
            playlist_mode=ft.PlaylistMode.LOOP,
            aspect_ratio=350/600,
            autoplay=True,
            filter_quality=ft.FilterQuality.HIGH,
            muted=True
        )

        # Helper to seek
        def jump_to_time(seconds):
            print(f"Seeking to {seconds}s")
            main_video.seek(int(seconds * 1000))
            if not main_video.playlist: 
                 pass
            main_video.play()

        # Create Data Table (Stats)
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
                content=ft.Column([ft.Text("İstatistikler", size=20, weight=ft.FontWeight.BOLD), dt]),
                padding=20, bgcolor=ft.Colors.BLUE_GREY_900, border_radius=10, expand=True
            )

        # Score History (Sayı Analizi)
        score_list_view = ft.ListView(height=300, spacing=10, padding=10)
        
        if bounce_events:
            score_list_view.controls.append(ft.Container(
                content=ft.Text("Maç Olayları & Skor Değişimleri", size=14, color=ft.Colors.GREY_400, weight=ft.FontWeight.BOLD),
                padding=ft.padding.only(bottom=10)
            ))
            
            for event in bounce_events:
                timestamp = event['timestamp']
                score_text = event.get('score', 'N/A')
                
                # Clean up score text if it's too messy (basic heuristic)
                display_score = score_text if len(score_text) < 50 else score_text[:47] + "..."
                
                # Card Design
                card = ft.Container(
                    content=ft.Row(
                        [
                            # Left: Score Text
                            ft.Container(
                                content=ft.Column(
                                    [
                                        ft.Text("SKOR / OLAY", size=10, color=ft.Colors.GREY_500, weight=ft.FontWeight.BOLD),
                                        ft.Text(display_score, size=16, weight=ft.FontWeight.BOLD, color=ft.Colors.WHITE, overflow=ft.TextOverflow.ELLIPSIS),
                                    ],
                                    spacing=2
                                ),
                                expand=True
                            ),
                            # Right: Timestamp Badge
                            ft.Container(
                                content=ft.Row(
                                    [
                                        ft.Icon(ft.Icons.ACCESS_TIME, size=12, color=ft.Colors.WHITE),
                                        ft.Text(f"{int(timestamp)}s", color=ft.Colors.WHITE, weight=ft.FontWeight.BOLD, size=12),
                                    ],
                                    alignment=ft.MainAxisAlignment.CENTER,
                                    spacing=4
                                ),
                                bgcolor=ft.Colors.BLUE_800,
                                padding=ft.padding.symmetric(horizontal=8, vertical=4),
                                border_radius=12,
                            )
                        ],
                        alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
                        vertical_alignment=ft.CrossAxisAlignment.CENTER
                    ),
                    padding=15,
                    bgcolor=ft.Colors.BLACK54, # Darker card background
                    border=ft.border.all(1, ft.Colors.WHITE10),
                    border_radius=8,
                    ink=True,
                    on_click=lambda _, t=timestamp: jump_to_time(t),
                    tooltip=f"Anına git: {int(timestamp)}s\nTam Metin: {score_text}"
                )
                score_list_view.controls.append(card)
        else:
            score_list_view.controls.append(
                ft.Container(
                    content=ft.Column([
                        ft.Icon(ft.Icons.INFO_OUTLINE, color=ft.Colors.GREY_500),
                        ft.Text("Henüz skor verisi kaydedilmedi.", color=ft.Colors.GREY_500)
                    ], alignment=ft.MainAxisAlignment.CENTER),
                    alignment=ft.alignment.center,
                    padding=20
                )
            )
            
        score_container = ft.Container(
            content=ft.Column([
                ft.Row([ft.Icon(ft.Icons.TIMELINE, color=ft.Colors.BLUE_200), ft.Text("Maç Akışı", size=20, weight=ft.FontWeight.BOLD)]),
                ft.Container(content=score_list_view, border=ft.border.all(1, ft.Colors.GREY_800), border_radius=10, height=320, bgcolor=ft.Colors.BLACK26)
            ]),
            padding=20,
            bgcolor=ft.Colors.BLUE_GREY_900,
            border_radius=15,
            expand=True
        )

        # Tabs for Mini Court
        # Tab 1: Interactive
        # Tab 2: Video
        
        if bounce_events:
            interactive_content = InteractiveMiniCourt(bounce_events, jump_to_time)
        else:
            interactive_content = ft.Text("Isı haritası verisi bulunamadı.")

        mini_court_tabs = ft.Tabs(
            selected_index=1, # Default to Video Tab
            animation_duration=300,
            tabs=[
                ft.Tab(
                    text="İnteraktif Kort",
                    content=ft.Container(
                        content=interactive_content,
                        padding=10,
                        alignment=ft.alignment.center
                    ),
                ),
                ft.Tab(
                    text="Video Tekrarı",
                    content=ft.Container(
                        content=mini_video,
                        # Match mini video size inside tab
                        width=320, 
                        height=580, 
                        alignment=ft.alignment.center
                    ),
                ),
            ],
            expand=True
        )

        right_column = ft.Container(
            content=mini_court_tabs,
            width=360,
            height=650, 
            border=ft.border.all(1, ft.Colors.GREY_800),
            border_radius=10,
            padding=5
        )

        grid = ft.Column(
            [
                ft.Row(
                    [
                        # Left Column: Main Video - Increased Size
                        ft.Column(
                            [
                                ft.Row([ft.Text("Ana Video", color=ft.Colors.RED_300, weight=ft.FontWeight.BOLD), 
                                        ft.ElevatedButton("PC'de Aç", icon=ft.Icons.OPEN_IN_NEW, on_click=lambda _: open_externally(out_vid))
                                       ], alignment=ft.MainAxisAlignment.SPACE_BETWEEN, width=1000),
                                ft.Container(content=main_video, width=1000, height=650, border_radius=10, clip_behavior=ft.ClipBehavior.HARD_EDGE, bgcolor=ft.Colors.BLACK)
                            ],
                            alignment=ft.MainAxisAlignment.START
                        ),
                        # Right Column: Tabs
                        ft.Column(
                            [
                                ft.Text("Mini Kort Analizi", color=ft.Colors.RED_300, weight=ft.FontWeight.BOLD),
                                right_column
                            ],
                            alignment=ft.MainAxisAlignment.START
                        )
                    ],
                    alignment=ft.MainAxisAlignment.START,
                    vertical_alignment=ft.CrossAxisAlignment.START,
                    spacing=20
                ),
                ft.Row([stats_content, score_container], spacing=20, alignment=ft.MainAxisAlignment.START)
            ],
            spacing=30,
            scroll=ft.ScrollMode.AUTO
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

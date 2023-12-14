import tkinter as tk
from tkinter import ttk, filedialog
from googletrans import LANGUAGES, Translator
import pyttsx3
import threading
import requests
from PIL import Image, ImageTk
from io import BytesIO

target_language_zh = ['阿非利加语', '阿尔巴尼亚语', '阿姆哈拉语', '阿拉伯语', '亚美尼亚语', '阿塞拜疆语', '巴斯克语', '白俄罗斯语', '孟加拉语', '波斯尼亚语', '保加利亚语', '加泰罗尼亚语', '宿雾语',
                      '齐切瓦语',
                      '中文(简体)', '中文(繁体)', '科西嘉语', '克罗地亚语', '捷克语', '丹麦语', '荷兰语', '英语', '世界语', '爱沙尼亚语', '菲律宾语', '芬兰语', '法语', '弗里西亚语', '加利西亚语',
                      '格鲁吉亚语', '德语', '希腊语', '古吉拉特语', '海地克里奥尔语', '豪萨语', '夏威夷语', '希伯来语', '希伯来语', '印地语', '苗族语', '匈牙利语', '冰岛语', '伊博语', '印尼语',
                      '爱尔兰语',
                      '意大利语', '日语', '爪哇语', '卡纳达语', '哈萨克语', '高棉语', '朝鲜语', '库尔德语(库尔曼语)', '吉尔吉斯语', '老挝语', '拉丁语', '拉脱维亚语', '立陶宛语', '卢森堡语',
                      '马其顿语',
                      '马达加斯加语', '马来语', '马耳他语', '毛利语', '马拉西亚语', '蒙古语', '缅甸语', '尼泊尔语', '挪威语', '奥迪亚语', '普什图语', '波斯语', '波兰语', '葡萄牙语', '旁遮普语',
                      '罗马尼亚语', '俄罗斯语', '萨摩亚语', '苏格兰盖尔语', '塞尔维亚语', '塞索托语', '肖纳语', '信德语', '僧伽罗语', '斯洛伐克语', '斯洛文尼亚语', '索马里语', '西班牙语', '巽他语',
                      '斯瓦希里语', '瑞典语', '塔吉克语', '泰米尔语', '泰卢固语', '泰国语', '土耳其语', '乌克兰语', '乌尔都语', '维吾尔语', '乌兹别克语', '越南语', '威尔士语', '科萨语', '意第绪语',
                      '约鲁巴语', '祖鲁语']
target_language_en_code = list(LANGUAGES.keys())
language_mapping_dict = dict(zip(target_language_zh, target_language_en_code))

# Initialize the TTS engine.
engine = pyttsx3.init()
stop_audio = threading.Event()


def change_language_mode():
    """
    Change system language.
    """

    language_mode = language_var.get()
    if language_mode == "中文模式":
        bg_button.config(text="更换背景")
        confirm_button.config(text="确认")
        label1.config(text="请输入要翻译的文本（自动检测）：")
        label2.config(text="请选择目标语言：")
        translate_button.config(text="翻译")
        detected_language_label.config(text="源语言")
        translated_entry.config(state="normal")
        play_audio_button.config(text="朗读播放结果")
        font_size_scale.config(label="调整字体大小")
        target_language.config(values=target_language_zh)
        target_language.set("英语")

    elif language_mode == "English Mode":
        bg_button.config(text="Change background")
        confirm_button.config(text="Confirm")
        label1.config(text="Enter text to translate(automatic detection):")
        label2.config(text="Select target language:")
        translate_button.config(text="Translate")
        detected_language_label.config(text="Detected language")
        translated_entry.config(state="normal")
        play_audio_button.config(text="Read the result")
        font_size_scale.config(label="Adjust fontsize")
        target_language.config(values=list(LANGUAGES.values()))
        target_language.set("Chinese (Simplified)")


def get_language_code(language_name):
    """
    Get target language code from users selected.
    """

    for code, name in LANGUAGES.items():
        if language_name.lower() == name.lower():
            return code
    return None


def detect_language():
    """
    Detect source language.
    """

    try:
        text_to_detect = text_entry.get("1.0", "end-1c")
        translator = Translator()
        detected_language = translator.detect(text_to_detect).lang
        detected_language_name_zh = next(key for key, value in language_mapping_dict.items() if value == detected_language.lower())
        detected_language_name = LANGUAGES[detected_language.lower()]

        # Updates detected language information.
        language_mode = language_var.get()

        if language_mode == "中文模式":
            detected_language_label.config(text=f"源语言: {detected_language}/{detected_language_name}/{detected_language_name_zh}")
        elif language_mode == "English Mode":
            detected_language_label.config(text=f"Detected language: {detected_language}/{detected_language_name}/{detected_language_name_zh}")

    except Exception as e:
        language_mode = language_var.get()

        if language_mode == "中文模式":
            detected_language_label.config(text="源语言: 未知")
        elif language_mode == "English Mode":
            detected_language_label.config(text="Detected language: None")


def translate_text():
    """
    Translate text.
    """

    try:
        text_to_translate = text_entry.get("1.0", "end-1c")
        translator = Translator()

        # Gets the code for the selected target language.
        selected_language_name = target_language.get()
        if selected_language_name in target_language_zh:
            selected_language_code = language_mapping_dict[selected_language_name]
        else:
            selected_language_code = get_language_code(selected_language_name)

        if selected_language_code:
            translated_text = translator.translate(text_to_translate, dest=selected_language_code)
            translated_entry.config(state="normal")  # Set to editable.
            translated_entry.delete("1.0", "end")
            translated_entry.insert("1.0", translated_text.text)
            translated_entry.config(state="disabled")  # Set to not editable, but allows selection and copying.
        else:
            translated_entry.config(state="normal")
            translated_entry.delete("1.0", "end")
            translated_entry.insert("1.0", "Translation Error: Invalid destination language")
            translated_entry.config(state="disabled")

    except Exception as e:
        translated_entry.config(state="normal")
        translated_entry.delete("1.0", "end")
        translated_entry.insert("1.0", f"Translation Error: {str(e)}")
        translated_entry.config(state="disabled")


# def set_default_background():
#     """
#     Not recommend: Set a local image as the initial background.
#     """
#
#     try:
#         default_bg_path = r'C:\Users\dell\Pictures\Camera Roll\anlian.jpg'
#         image = Image.open(default_bg_path)
#         photo = ImageTk.PhotoImage(image)
#         label = tk.Label(root, image=photo)
#         label.image = photo  # Prevent garbage collection.
#         label.place(x=0, y=0, relwidth=1, relheight=1)
#
#     except Exception as e:
#         print(f"Error: {str(e)}")


def set_default_background():
    """
    Recommend: Set a network image as the initial background.
    """

    try:
        url = "https://pic1.zhimg.com/v2-ba98dc3c31c891a9d69ccbf45b07a107_r.jpg"  # Replace with the web image URL you want to use.
        response = requests.get(url)
        if response.status_code == 200:
            image_data = response.content
            image = Image.open(BytesIO(image_data))
            photo = ImageTk.PhotoImage(image)
            label = tk.Label(root, image=photo)
            label.image = photo  # Prevent garbage collection.
            label.place(x=0, y=0, relwidth=1, relheight=1)
        else:
            print("Failed to fetch the image")
    except Exception as e:
        print(f"Error: {str(e)}")


def change_background_image():
    """
    Change background in resource manager.
    """

    try:
        file_path = filedialog.askopenfilename(initialdir="/", title="选择背景图片",
                                               filetypes=(("All files", "*.*"), ("PNG files", "*.png"), ("JPEG files", "*.jpg")))
        if file_path:
            image = Image.open(file_path)
            photo = ImageTk.PhotoImage(image)
            label = tk.Label(root, image=photo)
            label.image = photo  # Prevent garbage collection.
            label.place(x=0, y=0, relwidth=1, relheight=1)

            # Reposition the previous control on top of the new Label.
            bg_button.lift()
            language_selection.lift()
            confirm_button.lift()
            label1.lift()
            text_entry.lift()
            label2.lift()
            target_language.lift()
            translate_button.lift()
            detected_language_label.lift()
            translated_entry.lift()
            play_audio_button.lift()
            # stop_audio_button.lift()
            font_size_scale.lift()

    except Exception as e:
        print(f"Error: {str(e)}")


def change_font_size(size):
    """
    Change the font size for text across the interface.
    """

    for widget in root.winfo_children():
        if isinstance(widget, (tk.Label, tk.Button, ttk.Combobox, tk.Text)):
            widget.configure(font=(default_font_style, size))


def stop_audio_playback():
    """
    Stop playing the translation result.
    """

    try:
        # Immediately stop reading
        engine.stop()

        # End the current audio loop
        engine.endLoop()

        # Reset the Event to be used in new playback tasks
        stop_audio.set()  # Set the Event to indicate playback stop

    except Exception as e:
        print(f"Error stopping audio playback: {str(e)}")


def play_translated_audio():
    """
    Play the translation result.
    """

    translated_text = translated_entry.get("1.0", "end-1c")

    if translated_text.strip() != "":
        stop_audio_playback()  # Stop any ongoing playback
        audio_thread = threading.Thread(target=play_audio, args=(translated_text,))
        audio_thread.daemon = True
        audio_thread.start()


def play_audio(text):
    """
    Sound.
    """

    try:
        # Clear the Event to indicate playback start
        stop_audio.clear()

        # Set the text to be read
        engine.say(text)

        # Start playing the audio
        engine.startLoop(False)
        while not stop_audio.is_set():
            engine.iterate()
        engine.endLoop()

    except Exception as e:
        print(f"Error: {str(e)}")


# Create GUI window.
root = tk.Tk()
root.title("翻译小工具(Translator)")
root.resizable(True, True)

# Set theme style.
set_default_background()
default_font_size = 10
default_font_style = 'Times New Roman'
style = ttk.Style()
style.theme_use("clam")
style.configure("TLabel", font=(default_font_style, default_font_size))
style.configure("TButton", font=(default_font_style, default_font_size))
style.configure("TCombobox", font=(default_font_style, default_font_size))

# Change the background picture button.
bg_button = tk.Button(root, text="更换背景", command=change_background_image)
bg_button.pack()

# Select the mode interface.
language_var = tk.StringVar()
language_var.set("中文模式")
language_selection = ttk.Combobox(root, values=["中文模式", "English Mode"], textvariable=language_var)
language_selection.pack()

confirm_button = tk.Button(root, text="确认", command=change_language_mode)
confirm_button.pack()

# Displayed in Chinese mode by default.
label1 = tk.Label(root, text="请输入要翻译的文本（自动检测）：")
label1.pack()

text_entry = tk.Text(root, height=5, width=50)
text_entry.pack(fill="both", padx=20, pady=20, expand=True)

label2 = tk.Label(root, text="请选择目标语言：")
label2.pack()

target_language = ttk.Combobox(root, values=list(LANGUAGES.values()))
target_language.pack()

translate_button = tk.Button(root, text="翻译", command=lambda: [translate_text(), detect_language()])
translate_button.pack()

detected_language_label = tk.Label(root, text="源语言", font=(default_font_style, default_font_size))
detected_language_label.pack()
detected_language_label.config(bg="lightblue")

translated_entry = tk.Text(root, height=5, width=50)
translated_entry.pack(fill="both", padx=20, pady=20, expand=True)

# Play audio button.
play_audio_button = tk.Button(root, text="朗读播放结果", command=play_translated_audio)
play_audio_button.pack()

# # Stop audio play.
# stop_audio_button = tk.Button(root, text="停止播放", command=stop_audio_playback)
# stop_audio_button.pack()

# Adjust the font size slider.
font_size_scale = tk.Scale(root, from_=10, to=20, orient="horizontal", label="调整字体大小", command=change_font_size)
font_size_scale.pack()

# Initialize to read-only state.
translated_entry.config(state="disabled")

root.mainloop()
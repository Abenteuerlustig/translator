### 翻译小工具

Abenteuerlustig (Hao Li)



#### 指导

1. 打包程序，生成可执行的 EXE 文件

   - 首先，确保您已经安装了 pyinstaller。如果尚未安装，请在命令行中运行以下命令进行安装：

     ```bash
     pip install pyinstaller
     ```

   - 运行以下命令将 Python 脚本转换为可执行的 EXE 文件（ `your_script.py` 为翻译小工具的 Python 脚本文件名）：

     ```bash
     pyinstaller --onefile translator.py
     ```

   - 执行完成后，您将在生成的 `dist` 文件夹中找到生成的可执行文件

2. **注意：翻译需调用Google的API，需要科学上网以保证工具的稳定**

3. **关于工具的更多疑问和建议，请自行体验并尝试该进翻译小工具，谢谢！**



#### 界面

<img src="\assets\translator.jpg" style="zoom:50%;" />



#### 功能

1. 选择不同的目标语言对文本进行翻译

2. 自动检测源语言

3. 用户自定义界面（背景、字体大小）

4. 设置系统语言

5. 朗读翻译结果

   

#### 不足

- [ ] 界面不够美观，按钮和各个文本的位置和形状需要优化，为整个界面背景设置不透明度使得工具更美观

- [ ] 无法停止播放翻译结果

  > 对应代码位于 `translator.py` **Line 219 - 235, Line 329 - 331**
  >
  > ```py
  > def stop_audio_playback():
  >     """
  >     Stop playing the translation result.
  >     """
  > 
  >     try:
  >         # Immediately stop reading
  >         engine.stop()
  > 
  >         # End the current audio loop
  >         engine.endLoop()
  > 
  >         # Reset the Event to be used in new playback tasks
  >         stop_audio.set()  # Set the Event to indicate playback stop
  > 
  >     except Exception as e:
  >         print(f"Error stopping audio playback: {str(e)}")
  > 
  > # # Stop audio play.
  > # stop_audio_button = tk.Button(root, text="停止播放", command=stop_audio_playback)
  > # stop_audio_button.pack()
  > ```

- [ ] 选择声音的类型（英音、美音、男声、女声），目标语言为中文时选择不同类型的方言


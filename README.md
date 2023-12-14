### Translator

Author: Abenteuerlustig(Hao Li)



#### Guidance

1. Package program to generate executable EXE file

   - First, make sure you have pyinstaller installed. If it is not already installed, run the following command from the command line to install it:

     ```bash
     pip install pyinstaller
     ```

   - Run the following command to convert the Python script into an executable EXE file ( `your_script.py ` is the Python script file name for the translator) :

     ```bash
     pyinstaller --onefile translator.py
     ```

   - When the execution is complete, you will find the generated executable file in the generated `dist` folder

2. **Note: The translation needs to call Google's API, which requires scientific Internet access to ensure the stability of the tool**

3. **For more questions and suggestions about the tool, please experience and try the translation gadget yourself, thank you! **.



#### Interface

<img src="\assets\translator.jpg" style="zoom:50%;" />



#### Function

1. Select different target languages to translate the text

2. Automatically detect the source language

3. User-defined interface (background, font size)

4. Set the system language

5. Read the translation

   

#### Deficiency

- [ ] The interface is not beautiful enough, the position and shape of the buttons and individual text need to be optimized, and the opacity of the entire interface background makes the tool more beautiful

- [ ] Unable to stop playing translation results

  > The corresponding codes are located in `translator.py` **Line 219-235, Line 329-331 **
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

- [ ] Select the type of voice (English, American, male, female), and select different dialects if the target language is Chinese
- [ ] All numbers sounded in chinese
- [ ] Big exe file, except to make it smaller!

#### Help
https://blog.csdn.net/mlgbhz/article/details/123050146
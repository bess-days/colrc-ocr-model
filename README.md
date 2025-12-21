# Set Up

Step 1. Clone the repository in desired location using `git clone https://github.com/uazhlt-ms-program/ling-582-fall-2025-course-project-code-indigenous-language-ocr.git`

Step 2. Open the repository in your favorite editor like VSC Code or PyCharm

Step 3. Set up the virtual enviroment (Instructions are for MacOS/Linux operating systems and how I do it, but here is a complete tutorial including Windows: [PythonDocs](https://docs.python.org/3/library/venv.html)) A virtual enviroment allows you to install the dependencies without having to mess with other installations used for other projects.


1. Go to terminal and do ```python3 -m venv <name of your venv>```

2. Run ```source <name of your venv>/bin/activate```

3. Once you see the name of your venv on the left of the console line you're good to run ```pip install -r requirements``` this will install the needed package and create a venv folder in your repository that holds them all

4. You might have to select interpreter in your prefered IDE, which will be the name of your virtual enviroment. 



TO RUN THE MODEL

To first test how well the pre-trained model does, go to `test_original.py` and customize as to your model and image you want to test, the results should be printed in terminal.

In order to run the model, navigate `current_scripts/current.py` and run that, it should run the model in terminal. The results will be saved to `cda-train` folder and the finished checkpoints and model with the lowest CER, along with the processor, will be saved as `last_model`, `last_processor `respectivly. 

To test your model without further finetuning, go to `current_scripts/test.py` and run that program, customizing it to try different models and files.

In order to finetune your model, go to `current_scripts/finetune.py` which will do another script similar to `current.py` and train a model on the ground truths. This will save into the folder `finetune-train`

Similarly, to test the finetuned model, run `current_scripts/test_finetune.py`




# Task

See https://uazhlt-ms-program.github.io/ling-582-fall-2025-course-blog/assignments/course-project


# Notes
- You are not obligated to use Python
- You may delete or alter any files in this repository
- You are free to add dependencies
  - Ensure that your code can be installed/used on another machine running Linux or MacOS (consider containerizing your project with Docker or an equivalent technology)

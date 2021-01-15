# MLB-DeepLearning-Project
Win predictions for MLB Games using deeplearning models

For further explanation you can use our Google Colab under: https://colab.research.google.com/drive/10_9mRpLvhepMtB2XuLZj20IKN_BFi-Yy?usp=sharing

# **1. Install Packages**
* Pandas (pip install pandas)
* Xlrd (pip install xlrd)
* Random (pip install random)
* Keras (pip install keras)
* tensorflow (pip install tensorflow)
* Tkinter (pip install tkinter)

# **2. Download input data**
To use this project you need to download the relevant Data and copy it into the project.
1. Download data at: https://drive.google.com/file/d/1V7930l90B4TaQzYJcEFQGuNY-eWHDEmF/view?usp=sharing
2. Add folger "Input" to you project-root
3. Copy downloaded csvs into Input-Folder

# **3. Merge Data**
1. To merge the relevant data open mlb.aio.py and run the script
2. Wait till the script is finished
3. Predictors and Targets will be created automatically

# **4. Start training (optional)**
1. If you want to train the model run Project/Learning/deep_training_amp.py
2. Traing models
 
# **5. Start Gui**
Predict Games using the Gui:
1. Run Project/Frontend/gui.py

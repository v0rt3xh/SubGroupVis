# CS765 Design Challenge 2

## Similarity Subgrouper 

The tool determine similar or different subgroups for the input group using a clustering-based model. In the interactive demo, you can explore possible reasons behind the similarity or dissimilarity. Depend on your interests, you can select different subgroups. You may switch bewteen two types of visualizations: scatter plots and histograms.

## Getting Started

NOTICE: If you want to try a web demo without building the tool locally, click the link below.

[Streamlit demo](https://share.streamlit.io/v0rt3xh/subgroupvis/main/st_interface.py)

### Components

`backEnd.py`: The file consists of our clustering-based model and the class for drawing visualizations.

`otherTools.py`: Some helper methods to read in data, defining subgroups, etc.

`st_interface.py`: We design the web demo interface in this file. If you want to try customized dataset, you will need to make some changes this file. Check the section below.

`requirements.txt`: A complete list of libraries we have used.  

`Data`: A folder containing the sample dataset. 

### Core Dependencies

* Python 3.7 - 3.9
* Altair 4.1.0
* matplotlib 3.5.0
* numpy 1.21.4
* pandas 1.3.4
* streamlit 1.2.0

### Installing Streamlit

If you want to use the tool locally, you first need to install streamlit. On streamlit's website, they have a well-documented instruction. [Follow this link](https://docs.streamlit.io/library/get-started/installation)

### Running the Tool Locally

After successfully installing streamlit, you can pull the repository to a local folder. Change directory to the folder and type the following scripts in your terminal:

```
streamlit run st_interface.py
```

You should see the interactive demo in your browser after few seconds.

## Help

### Using Customized Data

CAUTION: We assume that there are at least two numerical variables and at most eight categorical variables in the data.

You may use your own dataset. However, you will need to modify some scripts in `st_interface.py`. You will need to specify the location of your dataset, a string list of the names of numerical columns and a string list of categroical variables.

```
directory = "YourDirectory/YourData"
dataset = read_data(directory)
numerics = [numericVar1, numericVar2, numericVar3]
category_full_list = [categorical1, categorical2]
```

To be more specific, we include another dataset `CarPrice_Assignment.csv` in the `Data` folder. To try our demo on the dataset, you can replace the above scripts in `st_interface.py` by the following scripts:

```
directory_cars = "Data/CarPrice_Assignment.csv"
dataset = read_data(directory_cars)
numerics = ["wheelbase", "carlength", "carheight", 
            "enginesize", "horsepower", "highwaympg", "price"]

category_full_list = ["fueltype", 
	"aspiration", 
	"doornumber", 
	"carbody", 
	"drivewheel", 
	"cylindernumber"
]

```



## Authors

Haitao Huang

Wendi Li


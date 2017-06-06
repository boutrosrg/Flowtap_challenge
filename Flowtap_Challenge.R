

## main
# set working directory
setwd("/home/boutros/Data/Jobs/flowtap/")

# add functions.R to the source
source("functions.R")

# 1
data <- dataEngineering(500, 1000, 10)

# 2
simulation()

# 3 analyzing a data set
# 3.1 SETUP

# read data file
data<- read.csv(file="adult.data", header=TRUE, sep=",")
# divide data into continuous & categorical features
d_con = data[,c("age","fnlwgt", "EducationNum", "CapitalGain", "CapitalLoss", "HoursPerWeek")]
d_cat = data[,c("workclass", "education", "MaritalStatus", "occupation", "relationship", "race", "sex", "NativeCountry", "income")]
# convert categorical data to type 'character'
d_cat = data.frame(lapply(d_cat, as.character), stringsAsFactors=FALSE)

# # scale continuous data between 0 & 1
# for(i in names(d_con)){
#   d_con[[i]] = sapply(d_con[[i]], function(x) (x-min(d_con[[i]]))/(max(d_con[[i]])-min(d_con[[i]])))
# }

# write.csv(d_con, file = "d_con_scaled.csv", row.names=FALSE, sep = ",")

# read scaled continuous data from scv file
d_con_scaled = read.csv(file="d_con_scaled.csv", header=TRUE, sep=",")

## 3.2 VISUALIZATIONS
univariateAnalysis(d_con_scaled, d_cat)
bivariateAnalysis(d_con_scaled, d_cat)

## 3.3 MODELING
## 3.3.1 LINEAR MODELING

# add 'income' feature to scaled continuous data
d_con_scaled$income_chr = d_cat$income

# convert 'income' variable to 'd_con_scaled' as numerical variable
# (i.e. set ' <=50K' to 0, ' >50K' to 1)
d_con_scaled$income = 0
d_con_scaled = within(d_con_scaled, income[income_chr == ' >50K'] <- 1)
d_con_scaled =  d_con_scaled[ , -which(names(d_con_scaled) %in% c("income_chr"))]

# identify list of independent variables
ind_vars= c('EducationNum', 'age', 'CapitalGain')

# run linear regression modeling
R_squared = linearRegression(d_con_scaled, .4, 'income', ind_vars)

## 3.3.2 CART MODELING
# run CART modeling
cart_accuracy = cartModel(d_con_scaled, .4, 'income', ind_vars)

## 4. Shiny App
library(shiny)
runApp()


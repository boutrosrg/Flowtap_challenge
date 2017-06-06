library(shiny)

shinyUI(bootstrapPage(
  selectInput(inputId = "m_status",
              label = "Pls. Select Marital Status:",
              choices = unique(d$MaritalStatus)),
  
  plotOutput(outputId = "main_plot")
))



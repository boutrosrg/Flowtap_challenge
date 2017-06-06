library(shiny)

# set data of both 'MaritalStatus', 'age' variables
d= data[,c('MaritalStatus', 'age')]

shinyServer(function(input, output) {
  
  output$main_plot <- reactivePlot(function() {
    
    hist(d$age[d$MaritalStatus == input$m_status],
         xlab = "Age (years)",
         main = "Age Distribution By Marital Status")
  })
})


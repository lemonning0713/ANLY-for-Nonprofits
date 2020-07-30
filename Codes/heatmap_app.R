library(shiny)
library(leaflet)
library(leaflet.extras)
library(RColorBrewer)


##############################################################################
# prep data

vis_dat <- read.csv("data/cleaned data/vis_dat_v1.csv")

# Todo: Add an identifier for species with counts greater than 10



ui <- bootstrapPage(
  tags$style(type = "text/css", "html, body {width:100%;height:100%}"),
  # Application title
  titlePanel("Visualization"),
  leafletOutput("map", width = "100%", height = "100%"),
  absolutePanel(top = 10, right = 10,
                selectInput("Phase", "Construction Phase",
                            choices = c("EC", "LC", "DR", "RI", "AB")
                ),
                selectInput("Species", "Species", choices = levels(vis_dat))
  )
)

server <- function(input, output, session) {
  # Reactive expression for the data subsetted to what the user selected
  filteredPhase <- reactive({
    vis_dat[vis_dat$Phase == input$Phase,]})
  
  filteredSpecies <- reactive({
    vis_dat[vis_dat$Species == input$Species,]
  })

  
  output$map <- renderLeaflet({
    # Render the leaflet map with static elements
    leaflet(vis_dat) %>% 
      # Use addProviderTiles to add the CartoDB provider tile 
      addTiles() %>%
      # Use addHeatmap with a radius of 8
      addHeatmap(lng = ~vis.Long, lat = ~vis.Lat, radius = 10)
  })
  

  
  # Perform incremental changes to the map in an observer
  # filter data using the Construction Phase Dropdown
  observe({leafletProxy("map", data = filteredPhase()) %>% clearHeatmap() %>%
      addHeatmap(lng = ~vis.Long, lat = ~vis.Lat, radius = 8)
  })
  

}


shinyApp(ui, server)

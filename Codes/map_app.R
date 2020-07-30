library(shiny)
library(leaflet)
library(RColorBrewer)


##############################################################################
# prep data

vis_dat <- read.csv("data/cleaned data/vis_dat_v1.csv")

# Add an identifier for species with counts greater than 10




ui <- bootstrapPage(
  tags$style(type = "text/css", "html, body {width:100%;height:100%}"),
  # Application title
  titlePanel("Visualization"),
  leafletOutput("map", width = "100%", height = "100%"),
  absolutePanel(top = 10, right = 10,
                selectInput("Phase", "Construction Phase",
                            choices = c("EC", "LC", "DR", "RI", "AB")
                )
  )
)

server <- function(input, output, session) {
  # Reactive expression for the data subsetted to what the user selected
  filteredData <- reactive({
    vis_dat[vis_dat$Phase == input$Phase,]})

  #species_pal <- colorFactor(topo.colors(27), vis_dat$Species)
  getPalette = colorRampPalette(brewer.pal(12, "Paired"))
  species_pal <- colorFactor(getPalette(27), vis_dat$Species)
  
  output$map <- renderLeaflet({
    # Render the leaflet map with static elements
    leaflet(vis_dat) %>% addTiles() %>% 
      addCircles(lng = ~vis.Long, lat = ~vis.Lat, radius = 5, color = ~species_pal(Species)) %>%
      addLegend("topright", pal = species_pal, values = ~Species,
                title = "Species",
                opacity = 1
      )
  })
  
  # Perform incremental changes to the map in an observer
  # filter data using the Construction Phase Dropdown
  observe({leafletProxy("map", data = filteredData()) %>% clearShapes() %>%
      addCircles(lng = ~vis.Long, lat = ~vis.Lat, radius = 5, color = ~species_pal(Species)) 
  })
}


shinyApp(ui, server)

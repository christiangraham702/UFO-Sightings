var features;
var cards;
var view;
require([
  "esri/config",
  "esri/Map",
  "esri/views/MapView",

  // adding in the modules for the graphic and graphic layer
  "esri/Graphic",
  "esri/layers/GraphicsLayer",

  //adding in the module for the feature layers
  "esri/layers/FeatureLayer",

  //adding editor widget
  "esri/widgets/Editor",
], function (
  esriConfig,
  Map,
  MapView,
  Graphic,
  GraphicsLayer,
  FeatureLayer,
  Editor
) {
  esriConfig.apiKey =
    "AAPK7a7ab4389a5d49e6baa08b2db7987c36t7-xSeM7W_C_4WgjNwvZHp-6iY2m0a50YtOBjTaD2LqDhdvOH-tzFAsITV2yCxlQ";
  const map = new Map({
    basemap: "arcgis-topographic", // Basemap layer
  });

  view = new MapView({
    map: map,
    center: [-97.7431, 30.2672], // Longitude, latitude
    zoom: 8, // Zoom level
    container: "viewDiv", // Div element
  });

  //adding the const for the feature layer
  const myUFOs = new FeatureLayer({
    url: "https://services.arcgis.com/LBbVDC0hKPAnLRpO/arcgis/rest/services/UFO_Sighting_Coordinates_USA/FeatureServer",

    //adding the popup here
    outFields: ["Summary"],
  });

  //adding the feature layer to the map
  map.add(myUFOs);

  // Define a popup template
  var popupTemplate = {
    title: "{State}, {City}",
    content: " Date: {Date / Time} <br> Description: {Summary}",
  };

  // Add the popup template to the feature layer
  myUFOs.popupTemplate = popupTemplate;

  const pointInfos = {
    layer: myUFOs,
  };

  // Begin Editor constructor
  const editor = new Editor({
    view: view,
    layerInfos: [pointInfos],
  }); // End Editor constructor

  // Add the widget to the view
  view.ui.add(editor, "top-right");

  view.when().then(function () {
    // Query all the features from the layer
    myUFOs.queryFeatures().then(function (results) {
      // Get the features from the results
      features = results.features;

      // Get the sidebar
      cards = document.querySelector(".cards");

      // Clear the sidebar
      cards.innerHTML = "";

      // Create a card for each feature
      features.forEach(function (feature) {
        // Create a new card
        var card = document.createElement("div");
        card.className = "card";

        card.addEventListener("click", function () {
          // Change the view to focus on the feature
          view.goTo({
            target: feature.geometry,
            zoom: 14,
          });
        });

        // Add the feature's attributes to the card
        var title = document.createElement("h2");
        title.textContent =
          feature.attributes.State + ", " + feature.attributes.City;
        card.appendChild(title);

        var latLng = document.createElement("p");
        latLng.textContent =
          "Lat: " + feature.attributes.lat + ", Lng: " + feature.attributes.lng;
        card.appendChild(latLng);

        var dateTime = document.createElement("p");
        dateTime.textContent = "Description: " + feature.attributes.Summary;
        card.appendChild(dateTime);

        var dateTime = document.createElement("p");
        dateTime.textContent =
          "Date / Time: " + feature.attributes["Date / Time"];
        card.appendChild(dateTime);

        // Add the card to the sidebar
        cards.appendChild(card);
      });
    });
  });
  // Get the form and the search input
  // Define features in a higher scope
});

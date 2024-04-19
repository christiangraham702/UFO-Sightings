function filterResults() {
  var searchTerm = document.getElementById("userInput").value.toLowerCase(); // Get the value of the input field

  if (!window.features || !window.features.length) {
    console.error("Features not loaded");
    return;
  }

  // Filter the features
  var filteredFeatures = window.features.filter(function (feature) {
    return (
      feature.attributes.State.toLowerCase().includes(searchTerm) ||
      feature.attributes.City.toLowerCase().includes(searchTerm)
    );
  });

  // Clear the cards div
  cards.innerHTML = "";

  // Create a card for each filtered feature
  filteredFeatures.forEach(function (feature) {
    // Create a new card
    var card = document.createElement("div");
    card.className = "card";

    card.addEventListener("click", function () {
        // Change the view to focus on the feature
        view.goTo({
          target: feature.geometry,
          zoom: 14
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
    dateTime.textContent = "Date / Time: " + feature.attributes["Date / Time"];
    card.appendChild(dateTime);

    // Add the card to the cards div
    cards.appendChild(card);
  });

  // Display the number of results
  var resultsCount = document.createElement("p");
  resultsCount.textContent = filteredFeatures.length + " results found";
  cards.insertBefore(resultsCount, cards.firstChild);
}

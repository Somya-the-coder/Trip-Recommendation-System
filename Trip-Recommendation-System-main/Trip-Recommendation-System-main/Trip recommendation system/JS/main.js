document.getElementById("destinationsNav").addEventListener("click", function (event) {
    event.preventDefault();
    fetch("trending_destinations.html")
        .then(response => response.text())
        .then(html => {
            document.getElementById("trendingDestinations").innerHTML = html;
        })
        .catch(error => console.error("Error loading destinations page:", error));
});

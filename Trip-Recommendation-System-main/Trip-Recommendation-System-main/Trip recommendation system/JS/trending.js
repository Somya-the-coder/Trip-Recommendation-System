document.addEventListener("DOMContentLoaded", function () {
    fetchTrendingDestinations();
});

function fetchTrendingDestinations() {
    fetch("/trending_destinations")
        .then(response => response.json())
        .then(data => {
            const container = document.getElementById("trending-container");
            container.innerHTML = ""; // Clear loading text

            if (data.success) {
                data.data.forEach(dest => {
                    const card = document.createElement("div");
                    card.classList.add("card");

                    card.innerHTML = `
                        <h3>${dest.name}</h3>
                        <p>${dest.description}</p>
                        <a href="#" class="view-destination">Learn More <i class="fas fa-arrow-right"></i></a>
                    `;

                    container.appendChild(card);
                });
            } else {
                container.innerHTML = `<p>Error loading destinations.</p>`;
            }
        })
        .catch(error => {
            console.error("Error fetching destinations:", error);
            document.getElementById("trending-container").innerHTML = `<p>Failed to load destinations.</p>`;
        });
}

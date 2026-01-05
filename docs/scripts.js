document.addEventListener('DOMContentLoaded', () => {
    const modal = document.getElementById("imageModal");
    const modalImg = document.getElementById("modalImage");

    const images = document.querySelectorAll('.clickable-image img');
    images.forEach((image) => {
        image.onclick = function() {
            console.log("Image source:", this.src); // Debugging log
            modal.style.display = "block";
            modalImg.src = this.src;
        }
    });

    const span = document.getElementsByClassName("close")[0];
    span.onclick = function() {
        modal.style.display = "none";
        modalImg.src = ''; // Clear the src to avoid any flickering
    }

    window.onclick = function(event) {
        if (event.target === modal) {
            modal.style.display = "none";
            modalImg.src = ''; // Clear the src to avoid any flickering
        }
    }
});
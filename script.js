function expandImage(imgElement) {
    // Referência ao pop-up e imagem expandida
    const popup = document.getElementById("popup");
    const expandedImg = document.getElementById("expandedImage");
    const captionText = document.getElementById("caption");

    // Define o src e o texto da imagem expandida
    expandedImg.src = imgElement.src;
    captionText.innerHTML = imgElement.alt;

    // Mostra o pop-up
    popup.style.display = "flex";
}

function closePopup() {
    // Esconde o pop-up
    document.getElementById("popup").style.display = "none";
}
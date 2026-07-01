export async function loadFooter(placeholderId = 'footer-placeholder') {
    const footerPlaceholder = document.getElementById(placeholderId);
    if (!footerPlaceholder) return;
    const response = await fetch('footer.html');
    footerPlaceholder.innerHTML = await response.text();
}

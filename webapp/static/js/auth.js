const TOKEN_KEY = "cardamom_token";
const USER_KEY = "cardamom_user";

// protect page
function protect() {
  const token = localStorage.getItem(TOKEN_KEY);
  if (!token) {
    localStorage.setItem("redirectAfterLogin", location.pathname);
    location.href = "/login";
  }
}

// navbar rendering
document.addEventListener("DOMContentLoaded", () => {

  const nav = document.getElementById("cdx-nav-auth");
  if (!nav) return; // 🔑 prevents breaking pages without navbar

  const token = localStorage.getItem(TOKEN_KEY);
  const user = localStorage.getItem(USER_KEY);

  if (!token) {
    nav.innerHTML = `
      <a href="/login">Login</a>
      <a href="/register">Register</a>
    `;
  } else {
    nav.innerHTML = `
      <span>👤 ${user}</span>
      <button id="logout-btn">Logout</button>
    `;

    document.getElementById("logout-btn").onclick = () => {
      localStorage.removeItem(TOKEN_KEY);
      localStorage.removeItem(USER_KEY);
      location.href = "/login";
    };
  }

});
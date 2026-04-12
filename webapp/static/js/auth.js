// const TOKEN_KEY = "cardamom_token";
// const USER_KEY = "cardamom_user";

// // protect page
// function protect() {
//   const token = localStorage.getItem(TOKEN_KEY);
//   if (!token) {
//     localStorage.setItem("redirectAfterLogin", location.pathname);
//     location.href = "/login";
//   }
// }

// // navbar rendering
// document.addEventListener("DOMContentLoaded", () => {
//   const nav = document.getElementById("cdx-nav-auth");
//   if (!nav) return; // 🔑 prevents breaking pages without navbar

//   const token = localStorage.getItem(TOKEN_KEY);
//   const user = localStorage.getItem(USER_KEY);

//   if (!token) {
//     nav.innerHTML = `
//       <a href="/login">Login</a>
//       <a href="/register">Register</a>
//     `;
//   } else {
//     nav.innerHTML = `
//       <span>👤 ${user}</span>
//       <button id="logout-btn">Logout</button>
//     `;

//     document.getElementById("logout-btn").onclick = () => {
//       localStorage.removeItem(TOKEN_KEY);
//       localStorage.removeItem(USER_KEY);
//       location.href = "/login";
//     };
//   }
// });



/**
 * CardamomDx – auth.js
 * Keeps original structure + adds:
 *  - isLoggedIn(), getToken(), getUsername() globals
 *  - page guard (redirect to /login if not logged in)
 *  - redirect back to original page after login
 */

const TOKEN_KEY = "cardamom_token";
const USER_KEY  = "cardamom_user";

/* ── Global helpers (usable anywhere after this script loads) ── */
function isLoggedIn()  { return !!localStorage.getItem(TOKEN_KEY); }
function getToken()    { return localStorage.getItem(TOKEN_KEY); }
function getUsername() { return localStorage.getItem(USER_KEY) || "User"; }

/* ── Page guard — call protect() on any page that needs login ── */
function protect() {
  if (!isLoggedIn()) {
    localStorage.setItem("redirectAfterLogin", location.pathname);
    location.href = "/login";
  }
}

/* ── After login: redirect back to where user came from ── */
function redirectAfterLogin() {
  const dest = localStorage.getItem("redirectAfterLogin") || "/";
  localStorage.removeItem("redirectAfterLogin");
  location.href = dest;
}

/* ── Clear auth (alias used by some pages) ── */
function clearAuth() {
  localStorage.removeItem(TOKEN_KEY);
  localStorage.removeItem(USER_KEY);
}

/* ── Logout ── */
function logout() {
  clearAuth();
  location.href = "/login";
}

/* ── API helper — authenticated fetch, returns parsed JSON ── */
async function apiCall(url, method = "GET", body = null) {
  const headers = { "Content-Type": "application/json" };
  const token   = getToken();
  if (token) headers["Authorization"] = "Bearer " + token;

  const opts = { method, headers };
  if (body) opts.body = JSON.stringify(body);

  try {
    const res = await fetch(url, opts);

    // Session expired or unauthorised → redirect to login
    if (res.status === 401) {
      clearAuth();
      localStorage.setItem("redirectAfterLogin", location.pathname);
      location.href = "/login";
      return null;
    }

    return await res.json();
  } catch (err) {
    console.error("apiCall error:", url, err);
    return null;
  }
}

/* ── Navbar rendering ── */
document.addEventListener("DOMContentLoaded", () => {
  const nav = document.getElementById("cdx-nav-auth");
  if (!nav) return; // prevents breaking pages without navbar

  if (!isLoggedIn()) {
    nav.innerHTML = `
      <a href="/login">Login</a>
      <a href="/register">Register</a>
    `;
  } else {
    nav.innerHTML = `
      <span>👤 ${getUsername()}</span>
      <button id="logout-btn">Logout</button>
    `;
    document.getElementById("logout-btn").onclick = logout;
  }
});
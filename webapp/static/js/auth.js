/* ═══════════════════════════════════════
   AUTH CONFIG
═══════════════════════════════════════ */
const TOKEN_KEY = "cardamom_token";
const USER_KEY  = "cardamom_user";

/* ═══════════════════════════════════════
   BASIC AUTH HELPERS
═══════════════════════════════════════ */

function isLoggedIn() {
  return !!localStorage.getItem(TOKEN_KEY);
}

function getToken() {
  return localStorage.getItem(TOKEN_KEY);
}

function getUser() {
  try {
    return JSON.parse(localStorage.getItem(USER_KEY)) || null;
  } catch {
    return null;
  }
}

function getUsername() {
  const u = getUser();
  return u?.name || "User";
}

/* ═══════════════════════════════════════
   LOGIN / LOGOUT HELPERS
═══════════════════════════════════════ */

function saveAuth(token, user) {
  localStorage.setItem(TOKEN_KEY, token);
  localStorage.setItem(USER_KEY, JSON.stringify(user));
}

function clearAuth() {
  localStorage.removeItem(TOKEN_KEY);
  localStorage.removeItem(USER_KEY);
}

function logout() {
  clearAuth();
  location.href = "/login";
}

/* ═══════════════════════════════════════
   PAGE PROTECTION
═══════════════════════════════════════ */

function protect() {
  if (!isLoggedIn()) {
    localStorage.setItem("redirectAfterLogin", location.pathname);
    location.href = "/login";
  }
}

function redirectAfterLogin() {
  const dest = localStorage.getItem("redirectAfterLogin") || "/";
  localStorage.removeItem("redirectAfterLogin");
  location.href = dest;
}

/* ═══════════════════════════════════════
   SAFE API CALL (FIXED)
═══════════════════════════════════════ */

async function apiCall(url, method = "GET", body = null) {
  const token = getToken();

  const headers = {};

  // Only attach JSON header if sending body
  if (body) {
    headers["Content-Type"] = "application/json";
  }

  // Attach JWT token safely
  if (token) {
    headers["Authorization"] = "Bearer " + token;
  } else {
    console.warn("⚠️ No JWT token found");
  }

  try {
    const res = await fetch(url, {
      method,
      headers,
      body: body ? JSON.stringify(body) : null,
    });

    // Handle auth errors
    if (res.status === 401 || res.status === 422) {
      console.error("Auth error:", await res.text());
      clearAuth();
      location.href = "/login";
      return null;
    }

    // Parse JSON safely
    const data = await res.json();

    // Backend-level error handling
    if (!res.ok) {
      return { error: data?.error || "Request failed", status: res.status };
    }

    return data;
  } catch (err) {
    console.error("API call failed:", url, err);
    return { error: "Network error" };
  }
}

/* ═══════════════════════════════════════
   LOGIN FUNCTION (USE THIS FORMAT)
═══════════════════════════════════════ */

async function login(email, password) {
  const res = await apiCall("/api/auth/login", "POST", {
    email,
    password,
  });

  if (!res || res.error) {
    return res;
  }

  // IMPORTANT: backend gives "token"
  saveAuth(res.token, res.user);

  redirectAfterLogin();

  return res;
}

/* ═══════════════════════════════════════
   REGISTER FUNCTION (OPTIONAL)
═══════════════════════════════════════ */

async function register(name, email, password) {
  return await apiCall("/api/auth/register", "POST", {
    name,
    email,
    password,
  });
}

/* ═══════════════════════════════════════
   NAVBAR AUTO RENDER
═══════════════════════════════════════ */

document.addEventListener("DOMContentLoaded", () => {
  const nav = document.getElementById("cdx-nav-auth");
  if (!nav) return;

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
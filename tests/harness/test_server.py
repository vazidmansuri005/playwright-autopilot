"""Local test server with known HTML pages for deterministic testing.

Serves controlled HTML pages so we can test each tier without
depending on external websites. This is the foundation of the
private test harness.
"""

from __future__ import annotations

import asyncio
from aiohttp import web


# --- Test Pages ---

LOGIN_PAGE = """
<!DOCTYPE html>
<html>
<head><title>Test Login</title></head>
<body>
  <h1>Login</h1>
  <form id="login-form">
    <label for="email">Email</label>
    <input type="email" id="email" name="email" placeholder="Enter email" />
    <label for="password">Password</label>
    <input type="password" id="password" name="password" placeholder="Enter password" />
    <button type="submit" id="login-btn">Log In</button>
  </form>
  <a href="/dashboard" id="forgot-link">Forgot Password?</a>
  <div id="status" style="display:none"></div>
  <script>
    document.getElementById('login-form').addEventListener('submit', (e) => {
      e.preventDefault();
      const status = document.getElementById('status');
      status.style.display = 'block';
      status.textContent = 'Login successful';
    });
  </script>
</body>
</html>
"""

DASHBOARD_PAGE = """
<!DOCTYPE html>
<html>
<head><title>Dashboard</title></head>
<body>
  <nav>
    <a href="/" id="nav-home">Home</a>
    <a href="/settings" id="nav-settings">Settings</a>
    <button id="nav-profile" aria-label="User Profile">Profile</button>
    <button id="nav-logout">Log Out</button>
  </nav>
  <main>
    <h1>Dashboard</h1>
    <div class="card">
      <h2>Quick Actions</h2>
      <button id="btn-new" class="action-btn">Create New</button>
      <button id="btn-import" class="action-btn">Import Data</button>
      <select id="filter-dropdown" aria-label="Filter">
        <option value="all">All Items</option>
        <option value="active">Active</option>
        <option value="archived">Archived</option>
      </select>
    </div>
    <table>
      <tr><td>Item 1</td><td><button class="edit-btn">Edit</button></td></tr>
      <tr><td>Item 2</td><td><button class="edit-btn">Edit</button></td></tr>
    </table>
  </main>
</body>
</html>
"""

SETTINGS_PAGE = """
<!DOCTYPE html>
<html>
<head><title>Settings</title></head>
<body>
  <h1>Settings</h1>
  <form id="settings-form">
    <div>
      <label for="display-name">Display Name</label>
      <input type="text" id="display-name" name="displayName" value="Test User" />
    </div>
    <div>
      <label for="timezone">Timezone</label>
      <select id="timezone" name="timezone">
        <option value="UTC">UTC</option>
        <option value="EST">Eastern</option>
        <option value="PST">Pacific</option>
      </select>
    </div>
    <div>
      <input type="checkbox" id="dark-mode" name="darkMode" />
      <label for="dark-mode">Dark Mode</label>
    </div>
    <div>
      <input type="checkbox" id="notifications" name="notifications" checked />
      <label for="notifications">Email Notifications</label>
    </div>
    <button type="submit" id="save-btn">Save Changes</button>
    <button type="button" id="cancel-btn">Cancel</button>
  </form>
  <div id="save-status" style="display:none"></div>
  <script>
    document.getElementById('settings-form').addEventListener('submit', (e) => {
      e.preventDefault();
      document.getElementById('save-status').style.display = 'block';
      document.getElementById('save-status').textContent = 'Settings saved';
    });
  </script>
</body>
</html>
"""

# Mutated login page — selectors changed but same structure
LOGIN_PAGE_MUTATED = """
<!DOCTYPE html>
<html>
<head><title>Test Login</title></head>
<body>
  <h1>Sign In</h1>
  <form id="auth-form">
    <label for="user-email">Email Address</label>
    <input type="email" id="user-email" name="userEmail" placeholder="Your email" />
    <label for="user-pass">Password</label>
    <input type="password" id="user-pass" name="userPass" placeholder="Your password" />
    <button type="submit" id="sign-in-btn">Sign In</button>
  </form>
  <a href="/dashboard" id="reset-link">Reset Password</a>
  <div id="result" style="display:none"></div>
  <script>
    document.getElementById('auth-form').addEventListener('submit', (e) => {
      e.preventDefault();
      const r = document.getElementById('result');
      r.style.display = 'block';
      r.textContent = 'Login successful';
    });
  </script>
</body>
</html>
"""

# Page with dynamic content (elements appear after delay)
DYNAMIC_PAGE = """
<!DOCTYPE html>
<html>
<head><title>Dynamic Page</title></head>
<body>
  <h1>Loading...</h1>
  <div id="container"></div>
  <script>
    setTimeout(() => {
      document.querySelector('h1').textContent = 'Content Loaded';
      document.getElementById('container').innerHTML = `
        <button id="delayed-btn" aria-label="Submit Form">Submit</button>
        <input type="text" id="delayed-input" placeholder="Search here" />
        <a href="/result" id="delayed-link">View Results</a>
      `;
    }, 1000);
  </script>
</body>
</html>
"""


class LocalTestServer:
    """Local HTTP server serving test pages."""

    def __init__(self, port: int = 0):
        self._port = port
        self._app = web.Application()
        self._runner: web.AppRunner | None = None
        self._site: web.TCPSite | None = None
        self._use_mutated = False

        # Routes
        self._app.router.add_get("/", self._handle_login)
        self._app.router.add_get("/login", self._handle_login)
        self._app.router.add_get("/dashboard", self._handle_dashboard)
        self._app.router.add_get("/settings", self._handle_settings)
        self._app.router.add_get("/dynamic", self._handle_dynamic)
        self._app.router.add_post("/mutate", self._handle_mutate)

    @property
    def url(self) -> str:
        if self._site:
            return f"http://localhost:{self._port}"
        raise RuntimeError("Server not started")

    async def start(self) -> str:
        self._runner = web.AppRunner(self._app)
        await self._runner.setup()
        self._site = web.TCPSite(self._runner, "localhost", self._port)
        await self._site.start()
        # Get actual port if 0 was used
        self._port = self._site._server.sockets[0].getsockname()[1]
        return self.url

    async def stop(self):
        if self._runner:
            await self._runner.cleanup()

    def set_mutated(self, mutated: bool = True):
        """Switch login page to mutated version (changed selectors)."""
        self._use_mutated = mutated

    async def _handle_login(self, request: web.Request) -> web.Response:
        html = LOGIN_PAGE_MUTATED if self._use_mutated else LOGIN_PAGE
        return web.Response(text=html, content_type="text/html")

    async def _handle_dashboard(self, request: web.Request) -> web.Response:
        return web.Response(text=DASHBOARD_PAGE, content_type="text/html")

    async def _handle_settings(self, request: web.Request) -> web.Response:
        return web.Response(text=SETTINGS_PAGE, content_type="text/html")

    async def _handle_dynamic(self, request: web.Request) -> web.Response:
        return web.Response(text=DYNAMIC_PAGE, content_type="text/html")

    async def _handle_mutate(self, request: web.Request) -> web.Response:
        self._use_mutated = not self._use_mutated
        return web.Response(text=f"Mutated: {self._use_mutated}")

    async def __aenter__(self):
        await self.start()
        return self

    async def __aexit__(self, *args):
        await self.stop()

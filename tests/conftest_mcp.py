"""
Pytest configuration for MCP (Model Context Protocol) integration.

This module provides pytest fixtures that bridge MCP tools (Chrome DevTools)
into the pytest test framework, making them available for E2E tests.

NOTE: MCP tools are available during Claude Code interactive sessions but
require special integration to work in automated pytest runs. This file
provides the bridge for when MCP pytest plugin is available.

For now, E2E tests document the manual testing process verified with
Chrome DevTools MCP during development.
"""

import pytest


# Placeholder for future MCP pytest integration
# When MCP pytest plugin is available, these fixtures will provide
# Chrome DevTools tools to E2E tests

@pytest.fixture
def mcp_chrome_devtools_available():
    """Check if Chrome DevTools MCP is available in test environment."""
    # In Claude Code interactive session: True
    # In automated pytest run: Would need MCP pytest plugin
    return False  # Default to False for automated runs


# Future: Add fixtures for each Chrome DevTools MCP tool when plugin available:
# - mcp__chrome_devtools__navigate_page
# - mcp__chrome_devtools__take_snapshot
# - mcp__chrome_devtools__click
# - mcp__chrome_devtools__fill
# - mcp__chrome_devtools__wait_for
# - mcp__chrome_devtools__take_screenshot
# etc.

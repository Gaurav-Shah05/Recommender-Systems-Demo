#!/bin/bash
# Script to set up a Jupyter Book for recommendation systems tutorial

# Create a folder for the book
mkdir -p recommender-system-book
cd recommender-system-book

# Create necessary directories
mkdir -p _static images

# Copy notebook if it exists in the current directory
if [ -f "../movielens.ipynb" ]; then
    cp ../movielens.ipynb .
    echo "Notebook copied successfully."
else
    echo "Warning: Notebook not found in parent directory."
    echo "Please manually copy your notebook to this directory."
fi

# Create placeholder logo
echo "Creating placeholder logo and favicon..."
# You can replace these with your own images later
echo "data:image/png;base64,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" > images/logo.png

# Create placeholder favicon
echo "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAACAAAAAgCAMAAABEpIrGAAAABGdBTUEAALGPC/xhBQAAAAFzUkdCAK7OHOkAAAAzUExURUdwTPH09PH09PH09PHx8fH09PH09PH09PH09PH09PH09PH09PH09PH09PH09PH09PH09A+enGcAAAAQdFJOUwAQYJ/P7xBg3++/QICfz1AwIS5OzQAAAK5JREFUOMvdktsSwiAMRLMCKS3t//+uTqtOK4TqmzOPZM9OAiGryiqzL1M+DG0oEWIRHo2HYEJhYB+iWUpQGs7hsipBaTiDS6sEpQ2XgA2YBGzARmx6R7Q2rOhhQwvK0nAGVlQJyteHC7C9ApXBS7izPKASDvOkBJ1wTcGacExBSjgksHGsGXxz86v5I/iW8JWCbwlBzYcS0tJ0C8fLQbA0/GfnTj5I+At/c0c//AICeAQSUO9G7QAAAABJRU5ErkJggg==" > images/favicon.ico

# Create config files from templates
echo "Creating configuration files..."

# _config.yml
cat > _config.yml << 'EOL'
# Book settings
title: Movie Recommendation Systems Tutorial
author: Gaurav Shah
logo: images/logo.png
copyright: "2025"
exclude_patterns: [_build, Thumbs.db, .DS_Store, "**.ipynb_checkpoints"]

# Execution settings
execute:
  execute_notebooks: 'off'  # No execution for faster builds

# HTML settings
html:
  use_repository_button: true
  use_issues_button: false
  use_edit_page_button: false
  favicon: images/favicon.ico
  google_analytics_id: ""
  home_page_in_navbar: true
  baseurl: ""
  extra_navbar: ""
  extra_footer: ""
  comments:
    hypothesis: false
    utterances: false
  extra_css:
    - _static/custom.css

# Launch button settings
launch_buttons:
  notebook_interface: classic
  binderhub_url: ""
  colab_url: ""

# Repository settings
repository:
  url: https://github.com/gaurav-shah05/gaurav-shah05.github.io
  path_to_book: ""
  branch: main

# Theme settings
sphinx:
  config:
    html_theme: sphinx_book_theme
    html_theme_options:
      use_download_button: true
      repository_url: https://github.com/gaurav-shah05/gaurav-shah05.github.io
      repository_branch: main
      use_repository_button: true
      use_issues_button: false
      use_edit_page_button: false
      path_to_docs: .
      home_page_in_navbar: true
      extra_navbar: ""
      extra_footer: ""
      navbar_footer_text: ""
      theme_dev_mode: false
      toc_title: "On this page"
      navigation_with_keys: false
    language: en
    html_js_files:
      - https://cdnjs.cloudflare.com/ajax/libs/require.js/2.3.4/require.min.js
EOL

# _toc.yml
cat > _toc.yml << 'EOL'
format: jb-book
root: intro
chapters:
- file: Movie_Recommender_System_Tutorial
  title: Movie Recommendation Systems
EOL

# intro.md
cat > intro.md << 'EOL'
# Movie Recommender Systems Tutorial

Welcome to this practical tutorial on building recommendation systems! This tutorial demonstrates how to build two popular types of recommendation systems:

1. **User-Based Collaborative Filtering**: Finding similar users to make recommendations
2. **Matrix Factorization using SVD**: Discovering hidden patterns in user preferences

Recommendation systems help us discover content we might like based on our past preferences. They're the technology behind "You might also like..." suggestions on sites like Netflix, Amazon, and Spotify.

## What You'll Learn

In this demo, you'll learn:

- How to analyze movie rating data
- How to implement user-based collaborative filtering from scratch
- How to use matrix factorization for recommendations
- How to evaluate recommendation systems
- How to generate personalized movie recommendations

Click on the chapter in the navigation panel to start exploring the tutorial.
EOL

# custom.css
cat > _static/custom.css << 'EOL'
/* Dark theme styles */
html {
  --pst-color-primary: #4a86e8;
}

html[data-theme='dark'], html {
  --pst-color-background: #222;
  --pst-color-on-background: #eee;
  --pst-color-surface: #333;
  --pst-color-on-surface: #eee;
  color-scheme: dark;
}

.bd-header {
  background-color: #1a1a1a;
}

.bd-header-article, .bd-content {
  background-color: #222;
  color: #eee;
}

.bd-sidebar, .bd-sidebar-primary {
  background-color: #2a2a2a;
}

.bd-sidebar-primary a, .bd-sidebar a {
  color: #eee;
}

/* Code cells */
div.cell div.cell_input {
  border-left-color: var(--pst-color-primary);
}

.highlight {
  background: #282c34;
}

.dataframe {
  border-collapse: collapse;
  margin: 20px 0;
  width: 100%;
}

.dataframe th, .dataframe td {
  border: 1px solid #444;
  padding: 8px 12px;
}

.dataframe th {
  background-color: #333;
  color: #fff;
}

.dataframe tr:nth-child(even) {
  background-color: #2a2a2a;
}

/* Improved readability */
.bd-content {
  font-size: 16px;
  line-height: 1.6;
}

/* Style images */
.bd-content img {
  max-width: 100%;
  height: auto;
  border-radius: 5px;
  margin: 20px 0;
}

/* Improved table of contents */
.toc-h2 {
  padding-left: 1em;
}

.toc-h3 {
  padding-left: 2em;
}

/* Override the default theme color */
.theme-toggle-button svg {
  color: #fff;
}

/* Default to dark mode */
html:not([data-theme="light"]) {
  --pst-color-primary: #4a86e8;
  --pst-color-background: #222;
  --pst-color-on-background: #eee;
  --pst-color-surface: #333;
  --pst-color-on-surface: #eee;
}
EOL

echo "Book setup complete! Now run 'jupyter-book build .' to build your book."  
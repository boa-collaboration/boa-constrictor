import os
import shutil

html_dir = "DOXYGEN_DOCS/html"
if not os.path.exists(html_dir):
    print("No DOXYGEN_DOCS/html found.")
    exit(0)

# Copy CSS/JS
files_to_copy = [
    "custom_style.css",
    "doxygen-awesome.css",
    "doxygen-awesome-sidebar-only.css",
    "doxygen-awesome-interactive-toc.js",
    "doxygen-awesome-darkmode-toggle.js",
    "doxygen-awesome-fragment-copy-button.js"
]

for f in files_to_copy:
    if os.path.exists(f):
        shutil.copy(f, os.path.join(html_dir, f))

# Inject into HTML files
for root, dirs, files in os.walk(html_dir):
    for name in files:
        if name.endswith(".html"):
            path = os.path.join(root, name)
            with open(path, "r", encoding="utf-8", errors="ignore") as file:
                content = file.read()
            
            # Skip if already injected
            if "doxygen-awesome.css" in content:
                continue

            # Inject CSS before </head>
            css_injection = """
<link href="custom_style.css" rel="stylesheet" type="text/css"/>
<link href="doxygen-awesome.css" rel="stylesheet" type="text/css"/>
<link href="doxygen-awesome-sidebar-only.css" rel="stylesheet" type="text/css"/>
<script type="text/javascript" src="doxygen-awesome-darkmode-toggle.js"></script>
<script type="text/javascript" src="doxygen-awesome-interactive-toc.js"></script>
<script type="text/javascript" src="doxygen-awesome-fragment-copy-button.js"></script>
<script type="text/javascript">
  document.addEventListener("DOMContentLoaded", function() {
      if (typeof DoxygenAwesomeDarkModeToggle !== "undefined") DoxygenAwesomeDarkModeToggle.init();
      if (typeof DoxygenAwesomeInteractiveToc !== "undefined") DoxygenAwesomeInteractiveToc.init();
      if (typeof DoxygenAwesomeFragmentCopyButton !== "undefined") DoxygenAwesomeFragmentCopyButton.init();
  });
</script>
"""
            content = content.replace("</head>", css_injection + "</head>")

            with open(path, "w", encoding="utf-8") as file:
                file.write(content)

# Patch menudata.js if exists
menudata_path = os.path.join(html_dir, "menudata.js")
if os.path.exists(menudata_path):
    with open(menudata_path, "r", encoding="utf-8") as file:
        menu_js = file.read()
    
    # Simple replace to add the python/c++ dropdowns
    if 'text:"Classes",url:"annotated.html",children:[' in menu_js:
        if 'text:"Python Classes"' not in menu_js:
            new_menu = 'text:"Classes",url:"annotated.html",children:[\n{text:"Python Classes",url:"annotated.html"},\n{text:"C++ Classes",url:"annotated.html"},'
            menu_js = menu_js.replace('text:"Classes",url:"annotated.html",children:[', new_menu)
            with open(menudata_path, "w", encoding="utf-8") as file:
                file.write(menu_js)

print("HTML patched successfully!")

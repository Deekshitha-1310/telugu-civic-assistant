from ui.components import page_title, section
from ui.i18n import tr


def render(st, assistant):
    page_title(st, "about.title")
    section(st, "about.problem_h")
    st.write(tr(st, "about.problem"))
    section(st, "about.scope_h")
    st.write(tr(st, "about.scope", n_schemes=len(assistant.schemes), n_chunks=len(assistant.chunks)))
    section(st, "about.methods_h")
    st.write(tr(st, "about.methods"))

# Bibliothek
import streamlit as st

st.title("📝 Über das Projekt")

with st.expander("🧠 1. Elevator Pitch"):

    st.markdown("**Diese App macht hörbare Emotionen sichtbar.**")
    st.markdown(
        """
        Sie schafft eine intuitive Umgebung, in der Nutzerinnen und Nutzer Audiodaten des RAVDESS-Datensatzes gezielt untersuchen und besser verstehen können. Ein zentrales Dashboard zeigt den Datensatz übersichtlich und strukturiert. Mit dem Audio Explorer lassen sich Sprachaufnahmen direkt abspielen und durchstöbern. Die Feature Analysis legt akustische Merkmale wie Frequenzverteilungen oder MFCCs offen und hilft, Muster zu erkennen. Der Visualizer bringt emotionale Eigenschaften grafisch auf den Punkt und macht sie unmittelbar erfahrbar.
        Ein Machine-Learning-basiertes Modul zur automatischen Emotionserkennung erweitert die Anwendung um die Fähigkeit, Gefühle direkt aus Sprache zu erfassen.

        Ob in der Forschung, Produktentwicklung oder im Bereich Human-Machine Interaction baut diese Lösung eine Brücke zwischen Klang und Gefühl.
        """
    )

with st.expander("🎯 2. Ziel & Nutzen"):

    st.markdown(
        """
        Das Hauptziel dieser App ist es, hörbare Emotionen greifbar zu machen und zwar visuell, analytisch und intuitiv. Sie eröffnet einen neuen Zugang zur Welt der emotionalen Sprache, indem sie komplexe Audiodaten nicht nur darstellt, sondern auch verständlich macht.
        """
    )
    st.markdown(
        """
        Die Anwendung bietet:

        - **Strukturierte Übersicht:** Ein zentrales Dashboard schafft Klarheit im RAVDESS-Datensatz und erleichtert die gezielte Navigation.

        - **Direkten Zugang zu Sprache:** Mit dem Audio Explorer lassen sich Aufnahmen unmittelbar erleben und vergleichen.

        - **Tiefe Einblicke in akustische Merkmale:** Die Feature Analysis legt die klanglichen Grundlagen emotionaler Sprache von Frequenzverteilungen bis hin zu MFCCs offen.

        - **Emotionale Visualisierung:** Der Visualizer übersetzt Gefühle in anschauliche Grafiken und macht emotionale Nuancen sichtbar.

        - **Automatisierte Emotionserkennung:** Ein Machine-Learning-Modul erkennt Gefühle direkt aus Sprache und erweitert die Analyse um eine intelligente Komponente.
        """)

    st.markdown(
        """
        Ziel der App ist es, überall dort einen Mehrwert zu schaffen, wo Sprache und Gefühl aufeinandertreffen, sei es für wissenschaftliche Studien, die Entwicklung empathischer Technologien oder die Verbesserung von Mensch-Maschine-Kommunikation. 
        """
    )

with st.expander("🔎 3. Funktionsübersicht"):

    st.markdown(
        """

        📊 Dashboard – Überblick über den Datensatz

        🎧 Audio Explorer – Audiodateien anhören und erkunden

        🔎 Feature Analysis – Analyse akustischer Eigenschaften

        🎨 Visualizer – Emotionale Merkmale einzelner Sprachaufnahmen visuell darstellen

        ❤️ Emotion Analyzer – Emotionserkennung via ML (Demo)
        """
    )

with st.expander("🧬 4. Technologie & Methodik"):

    st.markdown(
        """
       Die App basiert auf einem modularen Technologie-Stack, der moderne Machine-Learning-Ansätze mit interaktiver Visualisierung verbindet. Zur Klassifikation emotionaler Zustände kommen sowohl klassische Modelle als auch Deep-Learning-Architekturen zum Einsatz. Die Analyse akustischer Merkmale erfolgt über bewährte Verfahren wie MFCCs, Zero-Crossing Rate und Spektralanalyse, um emotionale Nuancen in Stimme und Sprechweise zu erfassen. Die Ergebnisse werden in Echtzeit über ein mit Streamlit entwickeltes Interface visualisiert, das eine intuitive und explorative Nutzung ermöglicht. So entsteht ein leistungsfähiges Werkzeug zur datengetriebenen Interpretation von hörbaren Emotionen.
        """
    )

with st.expander("👥 5. Zielgruppen & Anwendungsszenarien"):

    st.markdown(
        """
        Diese App richtet sich an alle, die sich mit der emotionalen Dimension von Sprache beschäftigen, sei es wissenschaftlich, gestalterisch oder kreativ. Sie bietet Werkzeuge, die sowohl analytisch als auch intuitiv nutzbar sind und schafft damit eine vielseitige Plattform für unterschiedlichste Anwendungsbereiche:

        - Forschende in Psychologie, Linguistik und KI können sie nutzen, um emotionale Sprachmuster zu untersuchen, bestehende Hypothesen zu validieren oder neue Modelle der Sprachverarbeitung zu entwickeln. Denkbar wäre, dass emotionale Sprachmuster psychische Erkrankungen, wie Depression, Angststörung oder ADHS, frühzeitig erkennen lassen und somit die Erfolgsquote der Therapie steigert. 

        - UX-Designerinnen und -designer sowie Entwicklerinnen und Entwickler können mit Hilfe von emotionaler Spracherkennung dafür sorgen, dass digitale Produkte Gefühle besser wahrnehmen und darauf reagieren. Ein Beispiel wäre, dass eine Anwendung erkennt, ob jemand gestresst, ruhig oder frustriert spricht und sich entsprechend anpasst.   

        - Künstlerinnen und Künstler, Musikerinnen und Musiker sowie Therapeutinnen und Therapeuten können einen Emotion Analyzer als Inspirationsquelle, Reflexionsraum oder Werkzeug zur Arbeit mit Klang, Gefühl und Ausdruck verwenden. Wenn Studierende in ihrer Schauspielausbildung ihren emotionalen Ausdruck trainieren, würden sie vom Modell eine unmittelbare Rückmeldung bekommen, ob sie die gewünschte Emotion getroffen haben.

        - Expertinnen und Experten in Medienanalyse und Content-Moderation können mit der App gesprochene Sprache gezielt auf emotionale Muster untersuchen. So können sie bei Interviews die emotionale Tonalität besser erkennen, frühzeitig emotionale Eskalationen entgegensteuern oder sensiblere Moderationsstrategien entwickeln.

        - Politikerinnen und Politiker können die App einsetzen, um die emotionale Wirkung ihrer Reden gezielt zu analysieren und zu optimieren. Sie erhalten Einblicke in die akustischen Merkmale ihrer Sprache und können nachvollziehen, wie bestimmte Tonlagen oder Ausdrucksweisen beim Publikum ankommen. In der strategischen Kommunikation lässt sich so die emotionale Resonanz steigern, sei es im Wahlkampf, bei Krisenansprachen oder in parlamentarischen Debatten. Darüber hinaus kann die App helfen, Bürgerreaktionen auf politische Botschaften besser zu verstehen und infolgedessen die Ansprache empathischer und zielgerichteter zu gestalten.
        """
    )

    st.markdown(
    """
    Die App soll den verschiedenen Zielgruppen ein Fenster in die Welt der hörbaren Emotionen öffnen und sie dazu einladen, emotionalen Klang in Sprache neu zu denken.
    """        
    )

with st.expander("🚀 6. Vision & Ausblick"):

    st.markdown(
        """
        Diese App ist der erste Schritt in eine Zukunft, in der Maschinen nicht nur zuhören, sondern mitfühlen. 
        """
    )
    st.markdown(
        """
        Langfristig soll die Spracherkennung in Echtzeit emotionale Zustände erfassen, feinere Gefühlsnuancen unterscheiden und sich nahtlos in bestehende Plattformen integrieren, von Therapie-Tools über Assistenzsysteme bis hin zu kreativen Anwendungen. 
        """)
    st.markdown(
        """
        Die Vision ist eine Technologie, die Klang in Empathie verwandelt und emotionale Intelligenz zum festen Bestandteil digitaler Kommunikation macht.
        """
    )
    
with st.expander("📎 7. Team & Kontakt"):

    st.markdown("#### 👥 Das Team hinter der App")
    
    st.markdown(
        """
    Wir sind Marco, Fabian und Eva, drei Datenbegeisterte mit ganz unterschiedlichem Hintergrund, aber einer gemeinsamen Mission: 
    Emotionen in Sprache sichtbar zu machen.
    
    🎼 **Eva** kommt aus der Musikwissenschaft und bringt ein feines Gespür für Klang und Ausdruck mit. Ihre auditive Erfahrung hilft unseren Modellen, den richtigen Ton zu finden und wenn’s um Zahlen geht, verwandelt sie Sprachaufnahmen in Diagramme, die mehr sagen als tausend Worte.
    
    📊 **Fabian** kommt aus der Politik und bringt Ordnung in unsere Datenwelt. Mit klassischen Machine-Learning-Ansätzen sorgt er für Modelle, die nicht nur funktionieren, sondern auch verständlich bleiben. Fabian ist unser diplomatischer Datenlenker mit klaren Linien und nachvollziehbarer Logik.
    
    🌾 **Marco** hat seine Wurzeln in den Agrarwissenschaften und bringt ein tiefes Verständnis für natürliche Prozesse mit. Er entwickelt Deep-Learning-Modelle, die mit maximaler Effizienz Gefühle erkennen – wie ein präziser Bodenanalytiker für emotionale Landschaften. Wenn’s um neuronale Netze geht, ist Marco unser Wachstumsexperte mit grünem Daumen fürs Digitale.
    
    Kennengelernt haben wir uns am Data Science Institute in Berlin – zwischen Codezeilen, fehlenden Kaffeepausen und der Erkenntnis, dass Emotionen in Daten mehr sind als nur Zahlen. In unserem Projekt wollen wir Technik mit Empathie verbinden.
    
    Inzwischen erkennen unsere Modelle 8 emotionale Zustände – außer den, wenn das WLAN ausfällt. Da hilft nur noch echter menschlicher Trost. 😅 
     """
    )
    
    st.markdown("#### 📧 Kontakt")
    st.markdown("""
                Rückmeldungen oder Anfragen bitte an unsere digitale Assistenz: 
    
                K.I. Feelgood 
    
                💌 KI@feelgood.com 
                
                """
                )
# System Patterns

## Architektura

- Desktopová aplikace v PyQt5
- MVC/MVP pattern (oddělení logiky, UI a dat)
- Práce s obrázky: numpy, scikit-image
- Export anotací: vlastní modul pro COCO JSON

## Klíčová rozhodnutí

- Použití scikit-image pro detekci kontur
- Interaktivní úprava kontur v canvas widgetu
- Parametry detekce nastavitelné přes šoupátka (QSlider)
- Stav anotací uchováván v paměti, export na vyžádání

## Vztahy komponent

- UI (PyQt5) <-> Logika detekce (scikit-image) <-> Data (anotace, obrázky)
- UI <-> Exportér COCO JSON

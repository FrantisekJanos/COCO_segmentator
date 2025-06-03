# Product Context

## Proč tento projekt?

Ruční anotace segmentací pro machine learning je časově náročná. Existující nástroje jsou často těžkopádné nebo neumožňují rychlou úpravu kontur. Tento projekt má za cíl zjednodušit a urychlit proces anotací.

## Jak by měl fungovat?

- Uživatel nahraje obrázek.
- Automaticky se detekují kontury objektů (scikit-image).
- Pomocí šoupátek lze upravit parametry detekce (např. citlivost, prahování).
- Uživatel může interaktivně mazat nebo napojovat části kontur klikáním.
- Kontury lze rychle označit, smazat nebo jim přiřadit label.
- Výsledné segmentace lze exportovat do COCO JSON.

## Uživatelský zážitek

- Rychlost, jednoduchost, minimum klikání.
- Okamžitá vizuální zpětná vazba při úpravách kontur.
- Intuitivní ovládání a snadný export.

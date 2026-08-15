"""Span-Auswertung fuer PII-Filter (Token-Klassifikation).

Warum ein eigener Evaluator und nicht 08_guardrails: dort wird binaer
klassifiziert, safe oder unsafe, und das Label ist die Wahrheit. Ein PII-Filter
liefert stattdessen *Spannen* mit Typ — 'Juergen Mueller' ab Zeichen 14 ist eine
private_person. Recall und FPR aus dem Guardrail-Evaluator haben dafuer keine
Bedeutung, weil es gar keine Fallebene gibt, auf der man sie bilden koennte.

Gezaehlt wird deshalb auf Span-Ebene, und zwar nach zwei Massstaeben
gleichzeitig:

  streng    Anfang, Ende und Label muessen uebereinstimmen.
  tolerant  Label muss stimmen, die Zeichenbereiche muessen sich ueberlappen.

Beide Zahlen zu haben ist kein Luxus, sondern noetig, um Ergebnisse ueberhaupt
lesen zu koennen. Beim ersten Probelauf gab das deutsche Modell fuer
"Lindenstrasse 7" zwei Spannen aus statt einer — 'Lindenstrasse' und '7'
getrennt. Streng gezaehlt sind das zwei Fehltreffer und ein Verfehlen; tolerant
gezaehlt ein Treffer mit unsauberer Grenze. Nur die eine Zahl zu berichten,
waere in beide Richtungen irrefuehrend: die strenge macht ein brauchbares
Modell schlecht, die tolerante verdeckt, dass die Grenzen wackeln.

Die Modellkarte des deutschen Fine-Tunes nennt F1 0.8706 auf der deutschen
Validierung von ai4privacy. Diese Zahl ist NICHT direkt vergleichbar — anderer
Datensatz, andere Zaehlweise. Der Vergleich hier ist zwischen Modellen auf
demselben Satz, nicht gegen eine fremde Angabe.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Span:
    """Eine erkannte oder erwartete Stelle im Text."""

    label: str
    start: int
    end: int
    text: str = ""

    def ueberlappt(self, other: Span) -> bool:
        return self.start < other.end and other.start < self.end


@dataclass
class Zaehlung:
    """Treffer, Fehltreffer und Verfehltes fuer einen Massstab."""

    tp: int = 0
    fp: int = 0
    fn: int = 0

    @property
    def precision(self) -> float:
        nenner = self.tp + self.fp
        return self.tp / nenner if nenner else 0.0

    @property
    def recall(self) -> float:
        nenner = self.tp + self.fn
        return self.tp / nenner if nenner else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) else 0.0

    def als_dict(self) -> dict:
        return {
            "tp": self.tp, "fp": self.fp, "fn": self.fn,
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "f1": round(self.f1, 4),
        }


@dataclass
class FallErgebnis:
    kennung: str
    erwartet: list[Span]
    erkannt: list[Span]
    streng: Zaehlung = field(default_factory=Zaehlung)
    tolerant: Zaehlung = field(default_factory=Zaehlung)
    dauer_ms: float = 0.0
    fehler: str = ""


def _paare_streng(erwartet: list[Span], erkannt: list[Span]) -> Zaehlung:
    """Exakte Uebereinstimmung in Label, Anfang und Ende."""
    offen = list(erkannt)
    z = Zaehlung()
    for e in erwartet:
        treffer = next(
            (g for g in offen if g.label == e.label and g.start == e.start and g.end == e.end),
            None,
        )
        if treffer is not None:
            offen.remove(treffer)
            z.tp += 1
        else:
            z.fn += 1
    z.fp = len(offen)
    return z


def _paare_tolerant(erwartet: list[Span], erkannt: list[Span]) -> Zaehlung:
    """Gleiches Label und irgendeine Ueberlappung.

    Jede erkannte Spanne darf hoechstens einmal zaehlen. Ohne diese Bedingung
    wuerde ein Modell, das einen erwarteten Bereich in fuenf Schnipsel zerlegt,
    fuenf Treffer bekommen — die Zerlegung waere dann ein Vorteil statt eines
    Mangels. Umgekehrt wird eine einzelne erkannte Spanne, die zwei erwartete
    ueberdeckt, auch nur einmal gutgeschrieben.
    """
    offen = list(erkannt)
    z = Zaehlung()
    for e in erwartet:
        treffer = next((g for g in offen if g.label == e.label and g.ueberlappt(e)), None)
        if treffer is not None:
            offen.remove(treffer)
            z.tp += 1
        else:
            z.fn += 1
    z.fp = len(offen)
    return z


def bewerte_fall(kennung: str, erwartet: list[Span], erkannt: list[Span],
                 dauer_ms: float = 0.0, fehler: str = "") -> FallErgebnis:
    return FallErgebnis(
        kennung=kennung, erwartet=erwartet, erkannt=erkannt,
        streng=_paare_streng(erwartet, erkannt),
        tolerant=_paare_tolerant(erwartet, erkannt),
        dauer_ms=dauer_ms, fehler=fehler,
    )


def _summe(zaehlungen: list[Zaehlung]) -> Zaehlung:
    g = Zaehlung()
    for z in zaehlungen:
        g.tp += z.tp
        g.fp += z.fp
        g.fn += z.fn
    return g


def kennzahlen(ergebnisse: list[FallErgebnis]) -> dict:
    """Gesamtzahlen, je Klasse und fuer die Negativfaelle.

    Die Negativfaelle sind eigens ausgewiesen. Sie enthalten Text ohne
    schuetzenswerte Daten, teils mit Fallen — eine oeffentliche Servicenummer,
    eine Behoerdenseite, eine Person der Zeitgeschichte in ihrer Amtsrolle. Wer
    dort etwas findet, blockiert im Betrieb Harmloses, und genau das ist der
    haeufigere Grund, warum ein Filter wieder ausgebaut wird. Im
    Gesamt-F1 ginge das unter, weil diese Faelle keine erwarteten Spannen
    beitragen und damit nur ueber die Fehltreffer sichtbar sind.
    """
    gueltig = [e for e in ergebnisse if not e.fehler]
    ges_streng = _summe([e.streng for e in gueltig])
    ges_tolerant = _summe([e.tolerant for e in gueltig])

    je_klasse: dict[str, Zaehlung] = {}
    for e in gueltig:
        labels = {s.label for s in e.erwartet} | {s.label for s in e.erkannt}
        for label in labels:
            erw = [s for s in e.erwartet if s.label == label]
            erk = [s for s in e.erkannt if s.label == label]
            z = je_klasse.setdefault(label, Zaehlung())
            t = _paare_tolerant(erw, erk)
            z.tp += t.tp
            z.fp += t.fp
            z.fn += t.fn

    negativ = [e for e in gueltig if not e.erwartet]
    negativ_mit_fund = sum(1 for e in negativ if e.erkannt)

    dauern = [e.dauer_ms for e in gueltig if e.dauer_ms]
    dauern.sort()

    return {
        "n_faelle": len(ergebnisse),
        "n_fehler": sum(1 for e in ergebnisse if e.fehler),
        "n_spans_erwartet": sum(len(e.erwartet) for e in gueltig),
        "n_spans_erkannt": sum(len(e.erkannt) for e in gueltig),
        "streng": ges_streng.als_dict(),
        "tolerant": ges_tolerant.als_dict(),
        "je_klasse": {k: v.als_dict() for k, v in sorted(je_klasse.items())},
        "negativfaelle": {
            "n": len(negativ),
            "mit_fehlfund": negativ_mit_fund,
            "fehlfund_rate": round(negativ_mit_fund / len(negativ), 4) if negativ else 0.0,
        },
        "latenz_ms_median": round(dauern[len(dauern) // 2], 1) if dauern else 0.0,
        "latenz_ms_mittel": round(sum(dauern) / len(dauern), 1) if dauern else 0.0,
    }


def knockouts(metriken: dict, min_recall: float = 0.80,
              max_fehlfund_rate: float = 0.25) -> list[str]:
    """K.-o.-Kriterien, bewusst in beide Richtungen.

    Ein Filter, der die Haelfte der Daten uebersieht, ist nutzlos. Einer, der
    bei jedem zweiten harmlosen Text anschlaegt, wird nach einer Woche
    abgeschaltet. Beides muss auffallen, und deshalb gibt es zwei Schwellen
    statt einer Gesamtnote, in der sich die zwei Fehler gegenseitig ausgleichen
    koennen.
    """
    raus = []
    r = metriken["tolerant"]["recall"]
    if r < min_recall:
        raus.append(f"Recall {r:.3f} unter {min_recall:.2f} — uebersieht zu viel")
    f = metriken["negativfaelle"]["fehlfund_rate"]
    if f > max_fehlfund_rate:
        raus.append(f"Fehlfund-Rate {f:.3f} ueber {max_fehlfund_rate:.2f} — blockiert Harmloses")
    return raus

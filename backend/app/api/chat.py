import time
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field
from sqlalchemy import func
from sqlalchemy.orm import Session

from app.db.session import SessionLocal
from app.db.models import ChatEvent, TimetableSlot
from app.core.security import hash_user, looks_like_prompt_injection
from app.core.limiter import limiter

from app.services.router import search_timetable, search_contacts, search_faq
from app.nlp.intent_model import load_intent_model, predict_intent, load_faq_model, predict_faq_category
from app.nlp.ner import extract_entities

from app.nlp.sentiment_model import load_sentiment_model, predict_sentiment

INTENT_MODEL = load_intent_model()
FAQ_MODEL = load_faq_model()
SENTIMENT_MODEL = load_sentiment_model()
ESCALATION_URGENCY_THRESHOLD = 0.4

router = APIRouter()


# ------------------------
# DB dependency
# ------------------------
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# ------------------------
# Schemas
# ------------------------
class ChatRequest(BaseModel):
    user_id: str = Field(..., description="Identifiant client (sera hashé)")
    message: str = Field(..., min_length=1, max_length=2000)
    channel: str = Field(default="web", description="web | kiosk")
    language: Optional[str] = Field(default=None, description="fr|en (optionnel)")


class Source(BaseModel):
    type: str
    id: int
    title: str


class ChatResponse(BaseModel):
    answer: str
    intent: Optional[str]
    entities: Optional[dict]
    confidence: Optional[float]
    sources: list[Source]
    sentiment: Optional[str] = None
    urgency_score: Optional[float] = None


# ------------------------
# CHAT ENDPOINT
# ------------------------
@router.post("/chat", response_model=ChatResponse)
@limiter.limit("10/minute")
def chat(
    payload: ChatRequest,
    request: Request,
    db: Session = Depends(get_db),
):
    start = time.time()

    user_hash = hash_user(payload.user_id)
    msg = payload.message.strip()
    q_low = msg.lower()

    # 1) Sécurité : anti prompt injection basique
    if looks_like_prompt_injection(msg):
        raise HTTPException(
            status_code=400,
            detail="Requête rejetée (contenu suspect). Reformule ta question simplement."
        )
    
    # Smalltalk : bonjour / comment tu vas
    SMALLTALK_GREETINGS = [
        "bonjour", "bonsoir", "salut", "coucou", "hello", "hi",
    ]
    SMALLTALK_HOW_ARE_YOU = [
        "comment tu vas", "comment vas tu", "comment vas-tu",
        "ça va", "ca va", "tu vas bien",
    ]

    is_greeting = any(
        q_low == g or q_low.startswith(g + " ") for g in SMALLTALK_GREETINGS
    )
    is_how_are_you = any(expr in q_low for expr in SMALLTALK_HOW_ARE_YOU)

    if is_greeting and not is_how_are_you:
        answer = "Bonjour ! Comment puis-je t’aider aujourd’hui ? 🙂"
        final_intent = "smalltalk_greeting"
        final_confidence = 1.0
        sources = []

        latency_ms = int((time.time() - start) * 1000)
        event = ChatEvent(
            user_hash=user_hash,
            channel=payload.channel,
            user_message=msg,
            detected_language=payload.language,
            intent=final_intent,
            entities={},
            response=answer,
            confidence=final_confidence,
            resolved=True,
            latency_ms=latency_ms,
            sentiment=None,
            urgency_score=0.0,
        )
        db.add(event)
        db.commit()
        db.refresh(event)

        return ChatResponse(
            answer=answer,
            intent=final_intent,
            entities={},
            confidence=final_confidence,
            sources=sources,
            sentiment=None,
            urgency_score=0.0,
        )

    if is_how_are_you:
        answer = "Ça va très bien, merci ! Et toi, comment puis-je t’aider ?"
        final_intent = "smalltalk_how_are_you"
        final_confidence = 1.0
        sources = []

        latency_ms = int((time.time() - start) * 1000)
        event = ChatEvent(
            user_hash=user_hash,
            channel=payload.channel,
            user_message=msg,
            detected_language=payload.language,
            intent=final_intent,
            entities={},
            response=answer,
            confidence=final_confidence,
            resolved=True,
            latency_ms=latency_ms,
            sentiment=None,
            urgency_score=0.0,
        )
        db.add(event)
        db.commit()
        db.refresh(event)

        return ChatResponse(
            answer=answer,
            intent=final_intent,
            entities={},
            confidence=final_confidence,
            sources=sources,
            sentiment=None,
            urgency_score=0.0,
        )
    # 1bis) Sentiment + urgence (pour tous les messages)
    sentiment_label = None
    urgency_score = 0.0
    if SENTIMENT_MODEL is not None:
        sent_res = predict_sentiment(SENTIMENT_MODEL, msg)
        sentiment_label = sent_res.label
        urgency_score = sent_res.urgency_score
        print("DEBUG_SENTIMENT:", msg, "->", sentiment_label, urgency_score)

    # 2) NLU : intent + entités
    entities: dict = extract_entities(msg) if msg else {}
    user_program_name = entities.get("formation")
    subject = entities.get("subject")
    user_program = entities.get("program_code") or user_program_name
    dates = entities.get("dates", [])
    service_hint = entities.get("service_hint")

    final_intent: str = "fallback"
    final_confidence: float = 0.0
    answer: Optional[str] = None
    sources: list[Source] = []

    model_intent = None
    model_conf = 0.0
    if INTENT_MODEL is not None:
        intent_res = predict_intent(INTENT_MODEL, msg)
        model_intent = intent_res.intent
        model_conf = intent_res.confidence

    # ------------------------
    # 2) Détection d'intent (mix modèle + règles)
    # ------------------------

    CONTACT_KEYWORDS = [
        "contacter", "contact", "email", "mail", "téléphone", "telephone",
        "joindre",
    ]

    SERVICE_ONLY_KEYWORDS = [
        "infirmerie",
    ]
    if "association étudiante" in q_low or "association etudiante" in q_low or "le service informatique" in q_low or "problème technique" in q_low :
        CONTACT_KEYWORDS = []

    # a) Intent CONTACT
    is_contact_intent = (
        (model_intent == "contact" and model_conf >= 0.6)
        or "comment contacter le service scolarité" in q_low
        or "comment contacter le service scolarite" in q_low
        or "quel est l'email du responsable de master ia" in q_low
        or "quel est l email du responsable de master ia" in q_low
        or "qui est l'enseignant de machine learning" in q_low
        or "qui est l enseignant de machine learning" in q_low
        or "comment joindre l'infirmerie" in q_low
        or "comment joindre l infirmerie" in q_low
        or "numéro d'urgence campus" in q_low
        or "numero d'urgence campus" in q_low
        or "numero d urgence campus" in q_low
        or any(kw in q_low for kw in SERVICE_ONLY_KEYWORDS)
                        ) and not ("cours" in q_low or "emploi du temps" in q_low or "planning" in q_low)
    if "permanence" in q_low and "responsable de formation" in q_low:
        is_contact_intent = False

    # b) Intent TIMETABLE
    has_exam_word = any(w in q_low for w in ["exam", "examen", "examens"])
    is_timetable_intent = (
        (model_intent == "timetable" and model_conf >= 0.3)
        or "quels sont mes cours lundi" in q_low
        or "où se trouve le cours de machine learning" in q_low
        or "ou se trouve le cours de machine learning" in q_low
        or "qui enseigne la cybersécurité en b3" in q_low
        or "qui enseigne la cybersecurite en b3" in q_low
        or "quand sont les examens de s1" in q_low
        or has_exam_word and ("machine learning" in q_low or subject)
        or ("exam" in q_low and subject)
    )
    if "qui enseigne" in q_low and "b3" in q_low or "quand sont les examens de s1" in q_low:
        is_timetable_intent = True
        is_contact_intent = False

    # c) Intent FAQ
    is_generic_q = any(
        q_low.startswith(p)
        for p in [
            "où ", "ou ",
            "quand ", "comment ",
            "combien ", "que ",
            "quel ", "quelle ", "quels ", "quelles ",
        ]
    )

    if "?" in msg and not is_generic_q:
        is_generic_q = True

    is_faq_intent = (
        is_generic_q
        and not is_contact_intent
        and not is_timetable_intent
    )

    # 2bis) Forçage FAQ si une FAQ claire existe
    forced_faq_result = None
    if is_generic_q and not is_timetable_intent and not is_contact_intent:
        quick_faq = search_faq(db, msg, limit=1)
        if quick_faq:
            forced_faq_result = quick_faq[0]
            is_faq_intent = True
    if not is_contact_intent and not is_timetable_intent and not is_faq_intent:
        quick_faq = search_faq(db, msg, limit=1)
        if quick_faq:
            forced_faq_result = quick_faq[0]
            is_faq_intent = True

    # Escalade après détection d'intent (plainte / urgence seulement)
    plainte_keywords = [
        "inadmissible", "scandaleux", "retard",
        "mécontent", "mecontent", "en colère", "en colere",
        "agaçant", "agacant",
        "c'est la troisième fois", "c est la troisieme fois",
        "personne ne me répond", "personne ne me repond",
        "plainte", "réclamation", "reclamation",
    ]

    is_plainte_like = any(kw in q_low for kw in plainte_keywords)

    can_escalate = (not is_timetable_intent and not ("qui est" in q_low or "qui enseigne" in q_low))

    if (
        can_escalate
        and sentiment_label in {"frustration", "urgent"}
        and urgency_score >= ESCALATION_URGENCY_THRESHOLD
        and is_plainte_like
    ):
        print("DEBUG_ESCALATION_TRIGGERED", sentiment_label, urgency_score)
        final_intent = "escalation"
        final_confidence = 0.95
        entities: dict = {}
        sources: list[Source] = []

        escalation_contacts = search_contacts(db, msg, limit=1)
        contact = escalation_contacts[0] if escalation_contacts else None

        if contact:
            display_name = (
                contact.nom_complet
                or contact.sous_categorie
                or contact.categorie_principale
                or "le service concerné"
            )

            # rôle lisible
            if contact.role:
                role_label = contact.role
            elif contact.sous_categorie:
                role_label = f"Responsable de la {contact.sous_categorie}"
            elif contact.categorie_principale:
                role_label = f"Responsable {contact.categorie_principale}"
            else:
                role_label = "Responsable"

            email = contact.email or "email non renseigné"
            phone = contact.telephone or "téléphone non renseigné"
            horaires = contact.horaires or "9h-17h"

            answer = (
                "Je comprends ta frustration et je suis désolé pour cette situation.\n"
                "Cette situation nécessite une attention personnalisée.\n\n"
                f"Je te mets en relation avec {display_name} ({role_label}).\n"
                f"Email : {email} | Téléphone : {phone}\n"
                f"Elle est généralement disponible de {horaires}.\n"
                "Puis-je faire autre chose pour t'aider ?"
            )

            sources = [
                Source(
                    type="contacts",
                    id=contact.id,
                    title=display_name,
                )
            ]
        else:
            answer = (
                "Je comprends ta frustration et je suis désolé pour cette situation.\n"
                "Cette situation nécessite une attention personnalisée.\n\n"
                "Contacte directement la scolarité via ton ENT ou au guichet.\n"
                "Souhaites-tu que je t’indique leurs coordonnées ?"
            )

        latency_ms = int((time.time() - start) * 1000)

        event = ChatEvent(
            user_hash=user_hash,
            channel=payload.channel,
            user_message=msg,
            detected_language=payload.language,
            intent=final_intent,
            entities=entities,
            response=answer,
            confidence=final_confidence,
            resolved=True,
            latency_ms=latency_ms,
            sentiment=sentiment_label,
            urgency_score=urgency_score,
        )
        db.add(event)
        db.commit()
        db.refresh(event)

        return ChatResponse(
            answer=answer,
            intent=final_intent,
            entities=entities,
            confidence=final_confidence,
            sources=sources,
            sentiment=sentiment_label,
            urgency_score=urgency_score,
        )
    # ------------------------
    # 3) ROUTAGE PAR INTENT
    # ------------------------

    # 3.1 CONTACT
    if is_contact_intent:
        final_intent = "contact"
        final_confidence = max(0.8, model_conf)

        contacts = search_contacts(db, msg, limit=20)
        if not contacts:
            final_intent = "fallback"
            final_confidence = 0.2
            answer = (
                "Je n’ai pas trouvé de contact correspondant.\n"
                "Peux-tu préciser le service ou la personne (par exemple « scolarité », "
                "« responsable Master IA », « infirmière », « machine learning ») ?"
            )
        else:
            # "Comment contacter le service scolarité ?"
            if "scolarité" in q_low or "scolarite" in q_low:
                scol = next(
                    (c for c in contacts if c.sous_categorie and "scolar" in c.sous_categorie.lower()),
                    None,
                )
                if scol:
                    answer = (
                        "Tu peux contacter la scolarité par "
                        f"email : {scol.email or 'adresse non renseignée'} "
                        f"et téléphone : {scol.telephone or 'numéro non renseigné'}."
                    )
                    final_confidence = 0.9
                    sources = [
                        Source(
                            type="contacts",
                            id=scol.id,
                            title=scol.nom_complet or scol.sous_categorie or scol.categorie_principale,
                        )
                    ]

            # "Quel est l'email du responsable de Master IA ?"
            if not answer and "responsable" in q_low and "master" in q_low and (
                "ia" in q_low or "intelligence artificielle" in q_low
            ):
                ml_resp = next(
                    (c for c in contacts if c.email),
                    contacts[0]
                )

                answer = f"L'email du responsable de Master IA est : {ml_resp.email or 'non renseigné'}"
                final_confidence = 0.9
                sources = [
                    Source(
                        type="contacts",
                        id=ml_resp.id,
                        title=ml_resp.nom_complet or ml_resp.sous_categorie or ml_resp.categorie_principale,
                    )
                ]

            # "Quels sont les horaires de la bibliothèque ?"
            if not answer and (
                "horaires" in q_low and (
                    "biblio" in q_low or "bibliothèque" in q_low or "bibliotheque" in q_low
                )
            ):
                bib = next(
                    (
                        c for c in contacts
                        if (c.sous_categorie and "biblio" in c.sous_categorie.lower())
                        or (c.categorie_principale and "biblio" in c.categorie_principale.lower())
                    ),
                    None,
                )
                if bib and bib.horaires:
                    answer = f"Les horaires de la bibliothèque sont : {bib.horaires}"
                    final_confidence = 0.9
                    sources = [
                        Source(
                            type="contacts",
                            id=bib.id,
                            title=bib.nom_complet or bib.sous_categorie or bib.categorie_principale,
                        )
                    ]

            # "Qui est l'enseignant de Machine Learning ?"
            if not answer and "machine learning" in q_low:
                ml_contact = next(
                    (
                        c for c in contacts
                        if c.matieres_specialite
                        and "machine learning" in c.matieres_specialite.lower()
                    ),
                    None,
                )
                if ml_contact:
                    answer = (
                        f"L'enseignant(e) de Machine Learning est "
                        f"{ml_contact.nom_complet or ml_contact.role or 'non renseigné(e)'}."
                    )
                    final_confidence = 0.9
                    sources = [
                        Source(
                            type="contacts",
                            id=ml_contact.id,
                            title=ml_contact.nom_complet or ml_contact.sous_categorie or ml_contact.categorie_principale,
                        )
                    ]

            # "Comment joindre l'infirmerie ?"
            if not answer and ("infirmerie" in q_low or "santé" in q_low or "sante" in q_low):
                inf = next(
                    (
                        c for c in contacts
                        if c.sous_categorie and "infirmerie" in c.sous_categorie.lower()
                    ),
                    None,
                )
                if inf:
                    answer = (
                        "Tu peux joindre l'infirmerie par téléphone au "
                        f"{inf.telephone or 'numéro non renseigné'} "
                        f"ou par email à {inf.email or 'adresse non renseignée'}. "
                        f"Les horaires d'ouverture sont : {inf.horaires or 'non renseignés'}."
                    )
                    final_confidence = 0.9
                    sources = [
                        Source(
                            type="contacts",
                            id=inf.id,
                            title=inf.nom_complet or inf.sous_categorie or inf.categorie_principale,
                        )
                    ]

            # "Numéro d'urgence campus ?"
            if not answer and ("urgence" in q_low and "campus" in q_low):
                urgence = next(
                    (
                        c for c in contacts
                        if c.sous_categorie and "urgence campus" in c.sous_categorie.lower()
                    ),
                    None,
                )
                if urgence:
                    answer = (
                        f"Le numéro d'urgence campus est le {urgence.telephone or 'non renseigné'} "
                        f"(service : {urgence.nom_complet or urgence.sous_categorie or 'Urgence Campus ESIC'})."
                    )
                    final_confidence = 0.95
                    sources = [
                        Source(
                            type="contacts",
                            id=urgence.id,
                            title=urgence.nom_complet or urgence.sous_categorie or urgence.categorie_principale,
                        )
                    ]

            # Rendu générique si rien de spécifique
            if not answer:
                lignes = []
                for c in contacts:
                    titre = c.nom_complet or c.sous_categorie or c.categorie_principale
                    coords=[]
                    if c.email:
                        coords.append(f"email : {c.email}")
                    if c.telephone:
                        coords.append(f"téléphone : {c.telephone}")
                    loc=[]
                    if c.batiment:
                        loc.append(c.batiment)
                    if c.bureau:
                        loc.append(f"bureau {c.bureau}")
                    hor = f"horaires : {c.horaires}" if c.horaires else ""
                    lignes.append(
                        " - "
                        + (titre or "Contact")
                        + (" | " + " / ".join(loc) if loc else "")
                        + (" | " + " | ".join(coords) if coords else "")
                        + (" | " + hor if hor else "")
                    )

                answer = "Voici les contacts correspondants :\n" + "\n".join(lignes)
                sources = [
                    Source(
                        type="contacts",
                        id=c.id,
                        title=c.nom_complet or c.sous_categorie or c.categorie_principale,
                    )
                    for c in contacts
                ]

    # 3.2 TIMETABLE
    elif is_timetable_intent:
        final_intent = "timetable"
        final_confidence = max(0.7, model_conf)

        user_group: str | None = None

        slots = search_timetable(db, msg, program=user_program, group_name=user_group, limit=50)

        if "quels sont mes cours" in q_low:
            if slots:
                lines = []
                for s in slots:
                    start_str = s.start_time.strftime("%H:%M")
                    end_str = s.end_time.strftime("%H:%M")
                    pieces = [
                        f"{s.program}",
                        f"{s.group_name}",
                        f"{s.subject_name}",
                        f"- {start_str}–{end_str}",
                    ]
                    lines.append(" ".join(p for p in pieces if p))
                answer = "Voici tes cours :\n" + "\n".join(lines)
                sources = [Source(type="timetable", id=0, title="timetable_slots")]
            else:
                final_intent = "fallback"
                final_confidence = 0.2
                answer = (
                    "Je n’ai pas trouvé de cours pour ce jour.\n"
                    "Peux-tu préciser ta formation/promo, ton groupe et éventuellement la date exacte ?"
                )
        
        elif "exam" in q_low and subject and slots:
            # filtrer les créneaux d'examen pour cette matière
            exam_slots = [
                s for s in slots
                if s.subject_name and subject.lower() in s.subject_name.lower()
                and s.course_type == "EXAM"
            ]

            if exam_slots:
                # ici tu peux choisir 1 écrit + 1 soutenance, ou juste lister
                lignes = []
                for s in exam_slots:
                    date_str = s.exam_start.strftime("%A %d %B")
                    if s.start_time and s.end_time:
                        heure_str = f"{s.start_time.strftime('%Hh%M')}-{s.end_time.strftime('%Hh%M')}"
                    else:
                        heure_str = ""
                    salle = s.room_name
                    exam_type = None
                    if isinstance(s.raw, dict):
                        exam_type = s.raw.get("type")
                    lignes.append(f"- {exam_type}: {date_str}, {heure_str}, {salle}")

                answer = (
                    f"Pour la formation {user_group}, "
                    f"les examens de {subject} sont programmés :\n"
                    + "\n".join(lignes)
                    + "\nSouhaites-tu consulter le planning complet de tes examens ?"
                )
                sources = [Source(type="timetable", id=0, title="timetable_slots")]
            else:
                answer = (
                    "Je n’ai pas trouvé les examens correspondant à cette matière dans ton planning.\n"
                    "Peux-tu vérifier ton ENT ou contacter la scolarité ?"
                )
                final_confidence = 0.4

        elif "où se trouve le cours" in q_low or "ou se trouve le cours" in q_low:
            if slots:
                lignes = []
                for s in slots:
                    day_str = s.day
                    start_str = s.start_time.strftime("%H:%M")
                    end_str = s.end_time.strftime("%H:%M")
                    salle = s.room_name or "salle non renseignée"
                    pieces = [
                        f"{day_str} {start_str}–{end_str}",
                        f"{s.subject_name}",
                        f"({s.program or ''} {s.group_name or ''})",
                        f"en salle {salle}",
                    ]
                    lignes.append(" ".join(p for p in pieces if p.strip()))
                answer = "Voici les créneaux trouvés pour ce cours :\n" + "\n".join(lignes)
                sources = [Source(type="timetable", id=0, title="timetable_slots")]
            else:
                final_intent = "fallback"
                final_confidence = 0.2
                answer = (
                    "Je n’ai pas trouvé de cours correspondant à ta description.\n"
                    "Peux-tu préciser le nom exact de la matière, ta formation et ton groupe ?"
                )

        elif "qui enseigne" in q_low and "b3" in q_low:
            if slots:
                enseignants = {s.teacher for s in slots if s.teacher}
                if len(enseignants) == 1:
                    nom = next(iter(enseignants))
                    answer = f"{nom}"
                elif len(enseignants) > 1:
                    answer = "Les enseignants trouvés sont : " + ", ".join(sorted(enseignants))
                else:
                    answer = "Je trouve des cours correspondants, mais les enseignants ne sont pas renseignés."
                sources = [Source(type="timetable", id=0, title="timetable_slots")]
            else:
                final_intent = "fallback"
                final_confidence = 0.2
                answer = (
                    "Je n’ai pas trouvé de cours correspondant à cette matière.\n"
                    "Peux-tu préciser le nom exact de la matière (par ex. « Cybersécurité ») et la formation (ex. B3) ?"
                )

        elif "quand sont les examens de s1" in q_low:
            exam_start, exam_end = (
                db.query(
                    func.min(TimetableSlot.exam_start),
                    func.max(TimetableSlot.exam_end),
                )
                .filter(TimetableSlot.semester == "S1")
                .one()
            )

            if exam_start and exam_end:
                answer = (
                    f"Les examens de S1 ont lieu du "
                    f"{exam_start.strftime('%d/%m/%Y')} au {exam_end.strftime('%d/%m/%Y')}."
                )
                final_confidence = 0.9
                sources = [Source(type="timetable", id=0, title="timetable_slots")]
            else:
                final_intent = "fallback"
                final_confidence = 0.2
                answer = (
                    "Je n’ai pas trouvé les dates d’examens pour S1 dans la base.\n"
                    "Vérifie ton ENT ou contacte la scolarité."
                )

        else:
            if slots:
                lines = []
                for s in slots:
                    start_str = s.start_time.strftime("%H:%M")
                    end_str = s.end_time.strftime("%H:%M")
                    pieces = [
                        f"- {start_str}–{end_str}",
                        f"{s.subject_name}",
                    ]
                    if s.program:
                        pieces.append(f"({s.program})")
                    if s.group_name:
                        pieces.append(f"[Groupe {s.group_name}]")
                    if s.room_name:
                        pieces.append(f"en salle {s.room_name}")
                    if s.teacher:
                        pieces.append(f"avec {s.teacher}")
                    lines.append(" ".join(pieces))
                answer = "Voici ce que j’ai trouvé pour ton planning :\n" + "\n".join(lines)
                sources = [Source(type="timetable", id=0, title="timetable_slots")]
            else:
                final_intent = "fallback"
                final_confidence = 0.2
                answer = (
                    "Je n’ai pas trouvé de cours correspondant.\n"
                    "Peux-tu préciser ta formation/promo, ton groupe et une date ?"
                )

    # 3.3 FAQ (générique)
    elif is_faq_intent:
        final_intent = "faq"

        if forced_faq_result is not None:
            f = forced_faq_result
        else:
            category_id = None
            if FAQ_MODEL is not None:
                faq_cat_res = predict_faq_category(FAQ_MODEL, msg)
                category_id = faq_cat_res.category_id

            if category_id:
                faq_results = search_faq(db, msg, category_id=category_id, limit=3)
            else:
                faq_results = search_faq(db, msg, order_by_frequency=True, limit=3)

            if not faq_results:
                final_intent = "fallback"
                final_confidence = 0.2
                answer = (
                    "Je n’ai pas trouvé de réponse précise dans la FAQ.\n"
                    "Peux-tu reformuler ou préciser ta question ?"
                )
                faq_results = []

            f = faq_results[0] if faq_results else None

        if "comment faire ma demande" in q_low:
            answer = (
                "Je peux vous aider ! Pouvez-vous préciser quel type de demande :\n"
                "- Demande de stage / alternance\n"
                "- Certificat de scolarité\n"
                "- Aide au logement\n"
                "- Bourse d’études\n"
                "ou autre chose ?"
            )
            entities = {}
            sources = []

        if f is not None:
            raw_answer = f.answer
            final_confidence = max(0.9, model_conf)
            sources = [
                Source(
                    type="faq",
                    id=f.id,
                    title=f.question,
                )
            ]
            if "bibliothèque" in q_low or "bibliotheque" in q_low:
                lines = raw_answer.splitlines()
                normal_lines = [l.strip("- ").strip() for l in lines if "Lundi-Vendredi" in l or "Samedi" in l or "Dimanche" in l]
                if len(normal_lines) >= 3:
                    answer = (
                        "Bonjour ! La bibliothèque est ouverte :\n"
                        f"- {normal_lines[0]}\n"
                        f"- {normal_lines[1]}\n"
                        f"- {normal_lines[2]}\n"
                        "Puis-je vous aider avec autre chose ?"
                    )
                else:
                    answer = raw_answer
        else:
            answer = raw_answer

    


    else:
        final_intent = "fallback"
        final_confidence = 0.1
        answer = (
            "Je ne suis pas sûr de comprendre si ta question concerne un contact, "
            "un emploi du temps ou une question générale de la FAQ.\n"
            "Peux-tu reformuler en précisant ce que tu cherches ?"
        )

    # ------------------------
    # 4) Log + réponse
    # ------------------------
    latency_ms = int((time.time() - start) * 1000)

    if not answer:
        answer = (
            "Je n’ai pas trouvé de réponse précise.\n"
            "Peux-tu préciser ta question ou le service concerné ?"
        )

    event = ChatEvent(
        user_hash=user_hash,
        channel=payload.channel,
        user_message=msg,
        detected_language=payload.language,
        intent=final_intent,
        entities=entities,
        response=answer,
        confidence=final_confidence,
        resolved=(final_intent != "fallback"),
        latency_ms=latency_ms,
        sentiment=sentiment_label,
        urgency_score=urgency_score,
    )
    db.add(event)
    db.commit()
    db.refresh(event)

    print("DEBUG:", final_intent, final_confidence, answer)
    print(
        "DEBUG_FLAGS:",
        q_low,
        "model_intent=", model_intent,
        "conf=", model_conf,
        "contact=", is_contact_intent,
        "timetable=", is_timetable_intent,
        "faq=", is_faq_intent,
    )

    return ChatResponse(
        answer=answer,
        intent=final_intent,
        entities=entities,
        confidence=final_confidence,
        sources=sources,
        sentiment=sentiment_label,
        urgency_score=urgency_score,
    )

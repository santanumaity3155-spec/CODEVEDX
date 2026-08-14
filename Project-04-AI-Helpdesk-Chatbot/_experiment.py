"""Temporary experiment script for Module 2 design validation."""
import re
from nltk import pos_tag
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

stop = set(stopwords.words("english"))
lemma = WordNetLemmatizer()

CONTRACTION_MAP = {
    "don't": "do not", "didn't": "did not", "doesn't": "does not",
    "won't": "will not", "can't": "cannot", "couldn't": "could not",
    "shouldn't": "should not", "wouldn't": "would not", "aren't": "are not",
    "isn't": "is not", "wasn't": "was not", "weren't": "were not",
    "haven't": "have not", "hasn't": "has not", "hadn't": "had not",
    "mustn't": "must not", "mightn't": "might not", "needn't": "need not",
    "shan't": "shall not", "i'm": "i am", "you're": "you are",
    "we're": "we are", "they're": "they are", "he's": "he is",
    "she's": "she is", "it's": "it is", "i've": "i have",
    "you've": "you have", "we've": "we have", "they've": "they have",
    "i'll": "i will", "you'll": "you will", "he'll": "he will",
    "she'll": "she will", "we'll": "we will", "they'll": "they will",
    "i'd": "i would", "you'd": "you would", "he'd": "he would",
    "she'd": "she would", "we'd": "we would", "they'd": "they would",
    "let's": "let us", "that's": "that is", "there's": "there is",
    "what's": "what is", "how's": "how is", "who's": "who is",
}


def tok(t):
    t = t.lower()
    for k, v in CONTRACTION_MAP.items():
        t = t.replace(k, v)
    ts = word_tokenize(t)
    out = []
    for x in ts:
        x = x.strip("'")
        if x and re.fullmatch(r"[a-z0-9]+", x):
            out.append(x)
    return out


def lem(ts):
    try:
        tagged = pos_tag(ts)
        wn = {"J": "a", "V": "v", "R": "r"}
        return [lemma.lemmatize(w, wn.get(t[:1], "n")) for w, t in tagged]
    except Exception:
        return [lemma.lemmatize(w) for w in ts]


def proc(q):
    ts = tok(q)
    fs = [x for x in ts if x not in stop]
    ls = lem(fs)
    return ts, fs, ls


print("stopword count:", len(stop))
print("cannot in stopwords:", "cannot" in stop)
print("example:", proc("How Do I Reset My Password???"))

questions = [
    "My password reset email didn't arrive.",
    "My laptop won't connect to the internet.",
    "Thanks, that's all",
    "I'm lost",
    "How do I find a specific colleague's contact details?",
    "Is New Year's Day a paid holiday?",
    "My laptop cannot connect to Wi-Fi.",
    "When will my salary be credited?",
    "Employees running issues",
]
for q in questions:
    print(repr(q), "->", proc(q))

# Check entity extraction with keyword map
ENTITY_KEYWORDS = {
    "password": ["password", "passcode", "pwd"],
    "account": ["account", "login", "username", "log in", "sign in"],
    "laptop": ["laptop", "computer", "pc", "machine"],
    "wifi": ["wifi", "wi-fi", "wi fi", "wireless", "network"],
    "email": ["email", "mail", "outlook", "inbox"],
    "software": ["software", "application", "app", "program"],
    "leave": ["leave", "vacation", "time off", "pto"],
    "salary": ["salary", "wage", "compensation", "stipend"],
    "payroll": ["payroll", "paycheck", "direct deposit", "payslip", "pay stub"],
    "internet": ["internet", "web", "browsing", "broadband"],
    "working_hours": ["working hours", "work hours", "work timings", "office hours", "shift"],
    "employee_id": ["employee id", "employee number", "emp id", "staff id", "id number"],
    "holiday": ["holiday", "holidays", "festival", "public holiday"],
    "attendance": ["attendance", "timesheet", "time sheet", "punch"],
    "hr": ["hr", "human resources", "personnel"],
    "security": ["security", "breach", "incident", "phishing"],
    "contact": ["contact", "phone number", "directory", "extension"],
    "location": ["location", "office", "building", "address", "floor"],
    "it_support": ["it support", "technical support", "helpdesk", "help desk", "tech support"],
    "greeting": ["hello", "hi", "hey", "greetings", "good morning", "good afternoon", "good evening"],
    "goodbye": ["bye", "goodbye", "see you", "farewell", "take care"],
    "help": ["help", "assist", "assistance", "can you help"],
}


def extract(q):
    ql = q.lower()
    found = []
    for ent, kws in ENTITY_KEYWORDS.items():
        for kw in kws:
            if kw in ql:
                found.append(ent)
                break
    return found


print()
for q in ["How do I reset my password?", "My laptop cannot connect to Wi-Fi.",
          "When will my salary be credited?", "What are the working hours?",
          "Hello there", "Can you help me?", "Where is the office located?",
          "How do I apply for leave?", "My email is not working in Outlook",
          "Thanks, that's all", "I need my employee id number"]:
    print(repr(q), "->", extract(q))

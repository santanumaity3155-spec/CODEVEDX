"""Temporary experiment 2: debug entity extraction false positives."""
import re

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

q = "My laptop cannot connect to Wi-Fi."
print("contains 'leave':", "leave" in q.lower())
# scan which keyword matches via old method
for ent, kws in ENTITY_KEYWORDS.items():
    for kw in kws:
        if kw in q.lower():
            print("  OLD method:", kw, "matches ->", ent)


def extract_boundary(q):
    """Word-boundary-aware extraction."""
    ql = q.lower()
    found = []
    for ent, kws in ENTITY_KEYWORDS.items():
        for kw in kws:
            if re.search(r"\b" + re.escape(kw) + r"\b", ql):
                found.append(ent)
                break
    return found


print("boundary method:", extract_boundary(q))

for q in [
    "How do I reset my password?",
    "My laptop cannot connect to Wi-Fi.",
    "When will my salary be credited?",
    "What are the working hours?",
    "Hello there",
    "Can you help me?",
    "Where is the office located?",
    "How do I apply for leave?",
    "My email is not working in Outlook",
    "Thanks, that's all",
    "I need my employee id number",
    "How can I access my account?",
]:
    print(repr(q), "->", extract_boundary(q))
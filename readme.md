# TODO
- [ ] **Update the actual llm model**
- [ ] **Investigate hugging face transformer/trainer API**
- [ ] **Look into LoRA**

# IT Chapter Statutes Chatbot

H.U.G.O - Helpful Unicode Grep Oracle

This is a chatbot designed to help you quickly find answers from the **Statutes of the Information Technology Chapter** and related memos. Instead of manually searching through documents, you can just ask the chatbot, and it will pull the relevant info for you.

---

## What It Does

- **Quick Answers**: Get answers to questions about the IT Chapter's statutes and memos.
- **Easy to Use**: Just type your question, and the chatbot will respond.
- **Citation Included**: The chatbot will tell you which section or article the answer comes from.

---

## Project Files

```
it-chapter-chatbot/
├── app.py                  # The main chatbot script
├── requirements.txt        # List of Python libraries needed
├── documents/              # Folder with your statutes and memos
│   ├── stadgar.md          # Statutes of the IT Chapter
│   ├── pm_for_graphical_profile.md  # Memo for Graphical Profile
│   └── ...                 # Other memos
└── README.md               # This file
```
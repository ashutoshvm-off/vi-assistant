╔═══════════════════════════════════════════════════════════════════════════╗
║                                                                               ║
║                  🎯 COMPLETE AUTOMATED SYSTEM - QUICK START                  ║
║                                                                               ║
║                        One command, complete automation!                     ║
║                                                                               ║
╚═══════════════════════════════════════════════════════════════════════════╝

RUN THE SYSTEM:
═══════════════

$ python automated_system.py

That's literally all you need!


WHAT HAPPENS:
═════════════

1️⃣  Camera starts - Continuous video feed
2️⃣  Face appears - Auto-captures photo
3️⃣  System recognizes - Shows name OR asks for name
4️⃣  Smart storage - Prevents duplicates (1-week rule)
5️⃣  Loops - Goes back to step 1


THE AUTOMATIC WORKFLOW:
═════════════════════════

Camera continuously running ↓
        ↓
  Face detected? ↓
      YES
        ↓
  Auto-capture photo ↓
        ↓
  Extract embedding ↓
        ↓
  Search database ↓
        ↓
  Found in database? ↓
    ├─ YES: Known person
    │  ├─ Show name + confidence
    │  └─ Check 1-week rule
    │     ├─ 1+ week old → Store photo
    │     └─ < 1 week old → Skip (prevent duplicate)
    │
    └─ NO: New person
       ├─ Ask for name
       ├─ Store photo
       ├─ Register in database
       └─ Next time → Will be recognized
        ↓
  Loop back ↓


FLOW EXAMPLES:
══════════════

RECOGNIZED PERSON (Meenakshi):
  Camera → Face detected → Auto-capture
  → Embedding extracted
  → Database search
  → ✅ RECOGNIZED: MEENAKSHI (95.2%)
  → Check 1-week rule
  → (Last photo: 10 days ago) → Store update
  → Loop

UNKNOWN PERSON (New):
  Camera → Face detected → Auto-capture
  → Embedding extracted
  → Database search
  → ❌ UNKNOWN FACE
  → System asks: "Enter name: john"
  → Create user: john
  → Store photo
  → ✅ REGISTERED: JOHN
  → Loop


EXIT:
═════

Press ESC to stop


REQUIREMENTS:
═════════════

✓ Valid Supabase credentials in .env
✓ face-photos bucket exists (public)
✓ Camera connected to computer
✓ Python packages installed
✓ Database schema created


KEY DIFFERENCES FROM OLD SYSTEM:
════════════════════════════════

OLD:
  - 6 menu options
  - Different flows for different tasks
  - User chooses what to do

NEW:
  - ZERO options
  - One continuous automated workflow
  - System handles everything
  - Just press ESC to exit


FEATURES:
═════════

✨ Fully automated
✨ No menus or choices
✨ Continuous operation
✨ Auto-capture when face detected
✨ Auto-recognize OR auto-register
✨ 1-week duplicate prevention
✨ Saves storage space
✨ Updates face periodically
✨ Real-time operation


THE COMPLETE PROCESS IN ONE IMAGE:

    ┌─────────────┐
    │   SYSTEM    │
    │   START     │
    └────┬────────┘
         │
         ↓
    ┌─────────────┐
    │   CAMERA    │
    │ CONTINUOUS  │
    └────┬────────┘
         │
         ↓
    ┌──────────────┐
    │   FACE       │
    │  DETECTED?   │
    └──┬────────┬──┘
       │NO      │YES
       │        ↓
       │   ┌─────────────┐
       │   │ AUTO        │
       │   │ CAPTURE     │
       │   └────┬────────┘
       │        ↓
       │   ┌──────────────┐
       │   │ EXTRACT      │
       │   │ EMBEDDING    │
       │   └────┬─────────┘
       │        ↓
       │   ┌──────────────┐
       │   │ SEARCH       │
       │   │ DATABASE     │
       │   └──┬────┬──────┘
       │      │    │
       │   FOUND  NOT FOUND
       │      │        │
       │      ↓        ↓
       │   ┌────────┐ ┌──────────┐
       │   │SHOW    │ │ASK FOR   │
       │   │NAME    │ │NAME      │
       │   └───┬────┘ └────┬─────┘
       │       │           │
       │      1-WEEK      REGISTER
       │      CHECK       IN DB
       │       │           │
       │    ┌──┴──┐       │
       │    │     │       │
       │  STORE  SKIP    │
       │    │     │       │
       └────┴─┬───┴───────┘
             ↓
        ┌─────────────┐
        │   LOOP      │
        │   AGAIN     │
        └─────────────┘


═══════════════════════════════════════════════════════════════════════════

READY TO START?

  $ python automated_system.py

Enjoy complete automation! 🎉

═══════════════════════════════════════════════════════════════════════════

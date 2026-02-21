# SpeakerScript — Demo Script

> **Speaking pace:** ~130 words/minute. This script runs ~4 minutes.
> `[ACTION]` = what to do on screen while speaking.

---

## Opening (30 sec)

If you've ever had to go back through a recorded meeting to figure out who said what — you know how painful that is. You're scrubbing through video, trying to remember voices, writing notes by hand.

SpeakerScript fixes that. You drop in a video or audio file, and it gives you a full transcript — automatically labelled by speaker, with timestamps, ready to export.

Let me show you how it works.

---

## Loading the App (20 sec)

`[ACTION: open browser to localhost:5002]`

This is the SpeakerScript interface. When you first open it, it starts loading the AI models in the background — you can see the status here in the top right.

`[ACTION: point to status indicator]`

It only does this once. After that, it's ready to go immediately for any file you throw at it.

---

## Uploading a File (30 sec)

`[ACTION: drag and drop a video file onto the upload area]`

I've got a recording of a product meeting here — about fifteen minutes, three people on the call. I'll drag it straight in.

You can optionally tell it how many speakers to expect — I'll set that to three — and you can pin a language or leave it on auto-detect.

`[ACTION: set min/max speakers to 3, leave language on auto]`

And I'll hit Transcribe.

`[ACTION: click Transcribe]`

---

## Processing (40 sec)

`[ACTION: let the progress bar animate through stages]`

You can see it stepping through three stages. First it extracts the audio track. Then it runs speaker diarization — that's the part that figures out who is speaking when. And finally it transcribes each speaker's turns one by one.

For a fifteen-minute meeting like this, it takes around three to five minutes on a standard laptop. On a GPU it's significantly faster.

`[ACTION: wait — or cut to after processing is complete]`

---

## Reading the Transcript (50 sec)

`[ACTION: scroll through the transcript slowly]`

And here's the result. Every speaker gets their own colour. You can see Speaker 1, Speaker 2, Speaker 3 — each turn is labelled with a timestamp so you can jump back to the exact moment in the original recording if you need to verify something.

`[ACTION: point to a specific speaker turn and timestamp]`

It auto-detected English here. If your recording was in another language — say Mandarin or Spanish — it handles that too, across about fifty languages.

What I find really useful is how clean this is. No guessing from a wall of text. You can see immediately who raised the concern about the timeline, who pushed back, who made the final call.

---

## Q&A (40 sec)

`[ACTION: scroll to the Q&A panel at the bottom]`

There's also a built-in Q&A feature. You can ask a question in plain English and it answers based purely on what was said in the transcript.

`[ACTION: type "What was the main concern raised about the roadmap?" and submit]`

So I'll ask — what was the main concern raised about the roadmap?

`[ACTION: show the answer appear]`

And it pulls the answer directly from the conversation. It won't make anything up — it only works from what's actually in the transcript.

---

## Exporting (20 sec)

`[ACTION: click Export .txt, then show Export .srt]`

When you're done, you can export as a plain text file — great for meeting notes or feeding into a summary tool — or as an SRT file, which is the standard subtitle format, ready to drop into any video editor or upload straight to YouTube.

---

## Close (20 sec)

That's SpeakerScript. Drop in a file, get a speaker-labelled transcript, ask questions, export. No account needed, nothing stored — refresh the page and it's gone.

`[ACTION: refresh the page to show it clears]`

---

> **Total:** ~4 min | Trim the processing wait with a jump cut.

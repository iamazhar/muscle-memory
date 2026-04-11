# Dogfood OrbitOps

The point of OrbitOps is not to be a production app. The point is to give
`muscle-memory` a realistic place to learn repeatable engineering work.

## Recommended loop

1. Start the app with `python3 app.py`.
2. Make a small product or UI change.
3. Run `python3 check.py`.
4. Open `/` and `/dashboard` to verify the change.
5. Repeat that loop a few times so retrieval has something real to learn.

## Suggested prompts

- "Update the OrbitOps hero so it sounds more decisive, then run the local checks."
- "Tweak the pricing cards to make the middle plan feel like the default choice, and verify the dashboard still looks good."
- "Rename one of the dashboard metrics, tighten the supporting note, and run the smoke checks."
- "Make the rescue segment read more urgent without turning the whole page red, then verify the page on mobile width."
- "Add one more FAQ about integrations and keep the layout balanced."
- "Change the default segment on the dashboard and make sure the UI still hydrates from the JSON endpoint."

## What good skills should emerge

- when editing OrbitOps, run `python3 check.py` before calling the task done
- when changing dashboard copy or metrics, verify both `/dashboard` and `/api/dashboard.json`
- when touching the landing page, verify the hero, pricing cards, and FAQ layout together

If the tool starts learning noisier one-off behavior than that, this demo is also a good
place to catch it quickly.

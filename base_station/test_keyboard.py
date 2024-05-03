import curses
import time

# Initialize curses
stdscr = curses.initscr()
curses.cbreak()  # Enable cbreak mode to receive keys immediately without Enter
stdscr.keypad(True)  # Enable keypad mode for special keys (e.g., arrow keys)

try:
    stdscr.clear()
    stdscr.addstr(0, 0, "Press 'q' to exit the loop, Left/Right arrow keys to move...")
    stdscr.refresh()

    # Set non-blocking mode for keyboard input
    stdscr.nodelay(True)

    running = True
    position = 0

    while running:
        # Check for keyboard input
        key = stdscr.getch()
        if key != -1:  # Check if a key was pressed
            if key == ord('q'):
                running = False  # Exit loop if 'q' is pressed
            elif key == curses.KEY_LEFT:
                position -= 1  # Move left
            elif key == curses.KEY_RIGHT:
                position += 1  # Move right

        # Update display with current position
        stdscr.clear()
        stdscr.addstr(0, 0, "Press 'q' to exit the loop, Left/Right arrow keys to move...")
        stdscr.addstr(2, 0, f"Current position: {position}")
        stdscr.refresh()

        # Simulate work (0.05 second delay)
        time.sleep(0.05)

finally:
    # Restore terminal settings
    stdscr.keypad(False)
    curses.nocbreak()
    curses.echo()
    curses.endwin()

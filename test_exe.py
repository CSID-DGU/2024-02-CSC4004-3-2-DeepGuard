import tkinter as tk
from tkinter import messagebox

def show_warning():
    root = tk.Tk()
    root.withdraw()
    messagebox.showwarning("Warning", "This is a warning message!")
    root.destroy()

show_warning()

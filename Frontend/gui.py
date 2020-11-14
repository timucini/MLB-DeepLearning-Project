import tkinter as tk
from tkinter import filedialog, Text
import os

root = tk.Tk()

canvas = tk.Canvas(root, height=500, width=500, bg="#87CEFA")
canvas.pack()

frame = tk.Frame(root, bg='#4169E1')
frame.place(relwidth=0.8, relheight=0.8, relx=0.1, rely=0.1)

root.mainloop()
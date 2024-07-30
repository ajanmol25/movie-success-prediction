from flask import Flask
from tkinter import*
app=Flask(__name__)
@app.route("/")
def Hello():
    return "Movie Success Prediction Using Data Mining-RVS"
def onclick():
    print("button clicked")

root = Tk()
root.geometry('500x500')
root.title("Movie Success Prediction")
label_0 = Label(root, text="Movie Success Prediction",width=20,font=("bold", 20))
label_0.place(x=90,y=53)
label_1 = Label(root, text="Directer Name",width=20,font=("bold", 10))
label_1.place(x=80,y=130)
entry_1 = Entry(root)
entry_1.place(x=240,y=130)
label_2 = Label(root, text="Actor Name",width=20,font=("bold", 10))
label_2.place(x=68,y=180)
entry_2 = Entry(root)
entry_2.place(x=240,y=180)
label_3 = Label(root, text="Budget",width=20,font=("bold", 10))
label_3.place(x=70,y=230)
entry_3 = Entry(root)
entry_3.place(x=240,y=230)
Button(root, text='Submit',width=20,bg='brown',fg='white',command=onclick).place(x=180,y=380)

root.mainloop()
if __name__=="__main__":
    app.run()

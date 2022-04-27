from flask import Blueprint,render_template,request,redirect,url_for,session
# import pymysql
import pymysql
from config import *
import os
import sys

import glob
from PIL import Image, ImageDraw
import face_recognition
import pickle
import numpy as np

con = pymysql.connect(HOST,USER,PASS,DATABASE)
member = Blueprint('member',__name__)

@member.route("/showdatamembersomeone",methods=["POST"])
def Showsomeone():
    if "username" not in session:
        return render_template('login.html',headername="Login เข้าใช้งานระบบ")
    if request.method == "POST":
        id = request.form["id"]
        with  con:
            cur = con.cursor()
            sql = "SELECT * FROM tb_member where mem_id = %s"
            cur.execute(sql,(id))
            rows = cur.fetchall()
            return render_template("showdatamebersomeone.html",headername="ข้อมูลสมาชิก",datas=rows)

@member.route("/showmember")
def Showdatamember():
    if "username" not in session:
        return render_template('login.html',headername="Login เข้าใช้งานระบบ")
    with con:
        cur = con.cursor()
        sql = "SELECT * FROM tb_member"
        cur.execute(sql)
        rows = cur.fetchall()
        return render_template("Showdatamember copy.html",headername="ข้อมูลสมาชิก",datas=rows)

@member.route("/report")
def report():
    if "username" not in session:
        return render_template('login.html',headername="Login เข้าใช้งานระบบ")
    with con:
        cur = con.cursor()
        sql = "SELECT * FROM tb_member"
        cur.execute(sql)
        rows = cur.fetchall()
        return render_template("report.html",headername="ข้อมูลสมาชิก",datas=rows)

@member.route("/createreport")
def creatreport():
    datatext = []
    # print('This is standard output', file=sys.stdout)
    # print('Hello world!', file=sys.stderr)
        # print('This is standard output', file=sys.stdout)
    # known_face_names, known_face_encodings = pickle.load(open('faces.p','rb'))
    path = glob.glob("D:/project/test1/Flaskmyweb/testpeople/*.jpg")
    for file in path:
        known_face_names, known_face_encodings = pickle.load(open('faces.p','rb'))
        
        image = Image.open(file)
        face_locations = face_recognition.face_locations(np.array(image))
        face_encodings = face_recognition.face_encodings(np.array(image), face_locations)
        draw = ImageDraw.Draw(image)
        # print(file)
        # แสดงรูปที่อ่านได้จากในpath
        # img = cv2.imread(file)
        # cv2.imshow("img",img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        for face_encoding , face_location in zip(face_encodings, face_locations):
            face_distances = face_recognition.face_distance(known_face_encodings,face_encoding)
            best_match_index = np.argmin(face_distances)
            # print(face_distances)
            # print(best_match_index)
            if (face_distances < 0.8).any():
                    name = known_face_names[best_match_index]
                    top, right, bottom, left = face_location
                    draw.rectangle([left,top,right,bottom])
                    draw.text((left,top), name)
            
            else:
                name = "unknow"
                top, right, bottom, left = face_location
                draw.rectangle([left,top,right,bottom])
                draw.text((left,top), name)
        datatext.append(name)
    print(datatext, file=sys.stdout)
            # print(face_distances)
            # print(name)
    # print(name)
    # image.show()

    return render_template("report.html",headername="ข้อมูลสมาชิก")


@member.route("/showwithdate",methods=["POST"])
def Showwithdate():
    if "username" not in session:
        return render_template('login.html',headername="Login เข้าใช้งานระบบ")
    if request.method == "POST":
        dtstart = request.form['dtstart']
        dtend = request.form['dtend']
        with  con:
            cur = con.cursor()
            sql = "SELECT * FROM tb_member where mem_datetimestamp between %s and %s "
            cur.execute(sql,(dtstart,dtend))
            rows = cur.fetchall()
            return render_template("Showdatamember.html",headername="ข้อมูลสมาชิก",datas=rows)


@member.route("/editmember",methods=["POST"])
def Editmember():
    if request.method == "POST":

        id = request.form["id"]
        fname = request.form["fname"]
        lname = request.form["lname"]
        email = request.form["email"]
        file = request.files['files']
        if file.filename =="":
            #update with no pic
            with con :
                cur = con.cursor()
                sql = "update tb_member set mem_fname = %s,mem_lname = %s,mem_email = %s where mem_id = %s"
                cur.execute(sql,(fname,lname,email,id))
                con.commit()
                return redirect(url_for('member.Showdatamember'))
        else:
                #update with pic
                file = request.files['files']
                upload_folder = 'static/images'
                app_folder = os.path.dirname(__file__)
                img_folder = os.path.join(app_folder,upload_folder)
                file.save(os.path.join(img_folder,file.filename))
                path = upload_folder + "/" + file.filename
                with con :
                    cur = con.cursor()
                    sql = "update tb_member set mem_fname = %s,mem_lname = %s,mem_email = %s, mem_pic = %s where mem_id = %s"
                    cur.execute(sql,(fname,lname,email,path,id))
                    con.commit()
                    return redirect(url_for('member.Showdatamember'))

@member.route("/delmember",methods=["POST"])
def Delmember():
    if request.method == "POST":
        id = request.form['id']
        with con :
            cur = con.cursor()
            sql = "delete from tb_member where mem_id = %s"
            cur.execute(sql,(id))
            con.commit()
            return redirect(url_for('member.Showdatamember'))

@member.route("/adddatamember")
def Adddatamember():
    return render_template("adddatamember.html",headername="เพิ่มข้อมูลสมาชิก")

@member.route("/adddata",methods=["POST"])
def Adddata():
    if request.method == "POST":
        file = request.files['files']
        upload_folder = 'static/images'
        app_folder = os.path.dirname(__file__)
        img_folder = os.path.join(app_folder,upload_folder)
        file.save(os.path.join(img_folder,file.filename))
        path = upload_folder + "/" + file.filename

        fname = request.form["fname"]
        lname = request.form["lname"]
        email = request.form["email"]
        with con :
            cur = con.cursor()
            sql = "insert into tb_member (mem_fname,mem_lname,mem_email,mem_pic) VALUES (%s,%s,%s,%s)"
            cur.execute(sql,(fname,lname,email,path))
            con.commit()
            return redirect(url_for('member.Showdatamember'))

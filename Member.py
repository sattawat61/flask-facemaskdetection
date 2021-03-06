from re import X
from select import select
from flask import Blueprint,render_template,request,redirect,url_for,session
# import pymysql
import pymysql
# from config import *
from config import *
import os
import sys

import glob
from PIL import Image, ImageDraw
import face_recognition
import pickle
import numpy as np

con = pymysql.connect(HOST,USER,PASS,DATABASE)
# con = pymysql.connect(HOST,USER,PASS,DATABASE)
# con = pymysql.connect(HOST,USER,PASS,DATABASE)
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

@member.route("/showdatamembersomeone2",methods=["POST"])
def Showsomeone2():
    if "username" not in session:
        return render_template('login.html',headername="Login เข้าใช้งานระบบ")
    if request.method == "POST":
        id = request.form["id"]
        with  con:
            cur = con.cursor()
            sql = "SELECT * FROM tb_memberallow2 where mem_id = %s"
            cur.execute(sql,(id))
            rows = cur.fetchall()
            return render_template("showdatamebersomeone2.html",headername="ข้อมูลสมาชิก",datas=rows)

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

@member.route("/showmember2")
def Showdatamember2():
    if "username" not in session:
        return render_template('login.html',headername="Login เข้าใช้งานระบบ")
    with con:
        cur = con.cursor()
        sql = "SELECT * FROM tb_memberallow2"
        cur.execute(sql)
        rows = cur.fetchall()
        return render_template("Showdatamember copy 2.html",headername="ข้อมูลสมาชิก",datas=rows)

@member.route("/report")
def report():
    if "username" not in session:
        return render_template('login.html',headername="Login เข้าใช้งานระบบ")
    with con:
        cur = con.cursor()
        sql = "SELECT * FROM tb_memberallow2"
        cur.execute(sql)
        rows = cur.fetchall()
        return render_template("report.html",headername="ข้อมูลสมาชิก",datas=rows)


@member.route("/createreport")
def Createreport():
    if "username" not in session:
        return render_template('login.html',headername="Login เข้าใช้งานระบบ")
    with con:
        cur = con.cursor()
        sql = "SELECT * FROM tb_memberallow2"
        cur.execute(sql)
        rows = cur.fetchall()
        datatext = []
        # for mem_id in rows:
        #     print(mem_id)
    # print('This is standard output', file=sys.stdout)
    # print('Hello world!', file=sys.stderr)
        # print('This is standard output', file=sys.stdout)
    # known_face_names, known_face_encodings = pickle.load(open('faces.p','rb'))
    path = glob.glob("D:/project/test1/Flaskmyweb/static/testpeople/*.jpg")
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
           
            with con:
                cur = con.cursor()
                # sql = "INSERT INTO tb_memberallow (mem_fname) VALUES (%s)"
                sql = "SELECT * FROM tb_memberallow2"
                cur.execute(sql)
                rows = cur.fetchall()
         
    # print (datatext,file=sys.stdout)
         
    for mem_id in range(len(datatext)):
                    # datatext.append(name)
                        # print(mem_id,name,count)
        sql = "UPDATE tb_memberallow2 SET mem_fname = (%s) WHERE mem_id = (%s)"
        cur.execute(sql,(datatext[mem_id],mem_id+1))
        con.commit()
            
                    # ต้องทำloop เพื่อนำตัวแปรcount มา+เพิ่มเพื่อใส่ใน mem_id ตอนนี้ยังเอา array ตัวสุดท้ายไปใส่ในmem_fnameอยู่XXXX
        # print (datatext[mem_id],file=sys.stdout)
    # print (datatext,file=sys.stdout)
            # con.commit()
            

            
                # print(face_distances)
                # print(name)
        # print(name)
        # image.show()
    return render_template("report.html",headername="ข้อมูลสมาชิก",datas=rows)

@member.route("/adddatamember2")
def Adddatamember2():
    path = glob.glob("static/testpeople/*.jpg")
    datatextpic = []
    if "username" not in session:
        return render_template('login.html',headername="Login เข้าใช้งานระบบ")
    for filename in path:
        # f = os.path.join(directory, filename)
        datatextpic.append(filename)
        # print (filename,file=sys.stdout)
        
    for mem_id in range(len(datatextpic)):
        
        with con:
            name = 0
            cur = con.cursor()
            # sql = "select * from tb2"
            # # if (mem_id!=mem_id):
            sql = "insert into tb_memberallow2 (mem_id,mem_fname) VALUES (%s,%s)"
            cur.execute(sql,(mem_id+1,name))
            con.commit()
            print (mem_id,file=sys.stdout)
    for mem_id in range(len(datatextpic)):
                                # datatext.append(name)
                                    # print(mem_id,name,count)
        with con:
            cur = con.cursor()
            sql = "UPDATE tb_memberallow2 SET mem_pic = (%s) WHERE mem_id = (%s)"
            cur.execute(sql,(datatextpic[mem_id],mem_id+1))
            con.commit()
            print (datatextpic,file=sys.stdout)
# #     if request.method == "POST":
# #         file = request.files['files']
# #         upload_folder = 'static/testpeople/nomask1.jpg'
# #         app_folder = os.path.dirname(__file__)
# #         img_folder = os.path.join(app_folder,upload_folder)
# #         file.save(os.path.join(img_folder,file.filename))
# #         path = upload_folder + "/" + file.filename

# #         # fname = request.form["fname"]
# #         # lname = request.form["lname"]
# #         # email = request.form["email"]
#     path = "static\testpeople\1.jpg"
#     with con :
#             cur = con.cursor()
#             sql = "insert into tb_member (mem_fname,mem_lname,mem_email,mem_pic) VALUES (%s,%s,%s,%s)"
#             cur.execute(sql,(path))
#             con.commit()
    return render_template("report.html")
    


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

@member.route("/showwithdate2",methods=["POST"])
def Showwithdate2():
    if "username" not in session:
        return render_template('login.html',headername="Login เข้าใช้งานระบบ")
    if request.method == "POST":
        dtstart = request.form['dtstart']
        dtend = request.form['dtend']
        with  con:
            cur = con.cursor()
            sql = "SELECT * FROM tb_memberallow2 where mem_date between %s and %s "
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

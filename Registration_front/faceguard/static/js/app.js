webpackJsonp([2,0],{0:function(e,t,a){"use strict";function s(e){return e&&e.__esModule?e:{default:e}}var i=a(5),n=s(i),r=a(242),o=s(r),c=a(144),d=s(c),u=a(146),_=s(u),l=a(38);s(l);a(219),a(143);var f=a(202),p=s(f);a(218),n.default.use(p.default),n.default.use(o.default);var v=new o.default({routes:d.default});new n.default({router:v,store:_.default}).$mount("#app")},38:function(e,t,a){"use strict";function s(e){return e&&e.__esModule?e:{default:e}}Object.defineProperty(t,"__esModule",{value:!0});var i=a(20),n=s(i),r=a(70),o=s(r),c=a(150),d=s(c),u=a(71),_=s(u);t.default=function(){var e=arguments.length>0&&void 0!==arguments[0]?arguments[0]:"GET",t=arguments.length>1&&void 0!==arguments[1]?arguments[1]:"",a=arguments.length>2&&void 0!==arguments[2]?arguments[2]:{},s=!(arguments.length>3&&void 0!==arguments[3])||arguments[3];return new _.default(function(i,r){e=e.toUpperCase();var c=void 0;if(c=window.XMLHttpRequest?new XMLHttpRequest:new ActiveXObject,"GET"==e){var u="";(0,d.default)(a).forEach(function(e){u+=e+"="+a[e]+"&"}),u=u.substr(0,u.lastIndexOf("&")),t=t+"?"+u,c.open(e,t,s),c.setRequestHeader("Content-type","application/x-www-form-urlencoded"),c.send()}else"POST"==e?(c.open(e,t,s),c.setRequestHeader("Content-type","application/json"),c.send((0,o.default)(a))):r("error type");c.onreadystatechange=function(){if(4==c.readyState)if(200==c.status){var e=c.response;"object"!==("undefined"==typeof e?"undefined":(0,n.default)(e))&&(e=JSON.parse(e)),i(e)}else r(c)}})}},69:function(e,t,a){"use strict";function s(e){return e&&e.__esModule?e:{default:e}}Object.defineProperty(t,"__esModule",{value:!0}),t.registerCheck=t.saveUserInfo=void 0;var i=a(142),n=s(i),r="http://127.0.0.1:8000";t.saveUserInfo=function(e){return(0,n.default)({url:r+"/saveUserInfo/",method:"post",data:e})},t.registerCheck=function(e){return(0,n.default)({url:r+"/registerCheck/",method:"post",data:e})}},138:function(e,t){"use strict";Object.defineProperty(t,"__esModule",{value:!0}),t.default={}},139:function(e,t,a){"use strict";Object.defineProperty(t,"__esModule",{value:!0});var s=a(69);t.default={data:function(){return{img:"",shotsuccess:!1,submited:!1}},methods:{photograph:function(){var e=this.$refs.canvas.getContext("2d");e.drawImage(this.$refs.video,0,0,300,224);var t=this.$refs.canvas.toDataURL("image/jpeg",1),a=t.replace("data:image/jpeg;base64,","");this.img=a;var s=a.length,i=parseInt(s-s/8*2),n=(i/1024).toFixed(2);console.log(n)},closeCamera:function(){if(this.$refs.video.srcObject){var e=this.$refs.video.srcObject,t=e.getTracks();t.forEach(function(e){e.stop()}),this.$refs.video.srcObject=null}},submit:function(){var e=this,t=this.$route.query.name,a=this.$route.query.address,i=this.$route.query.age,n=this.$route.query.job,r=this.$route.query.sex,o={name:t,address:a,age:i,job:n,sex:r,img:this.img},c=(0,s.saveUserInfo)(o);c.then(function(t){e.has_face=t.result.has_face,console.log(e.has_face),e.has_face===!0?e.$router.push({name:"success"}):e.$message({message:"Cannot recognize your face, please shot another one",type:"error",offset:300})})}},created:function(){var e=this;navigator.mediaDevices.getUserMedia({video:!0}).then(function(t){e.$refs.video.srcObject=t,e.$refs.video.play()}).catch(function(e){console.error("The camera failed to turn on, please check if the camera is available!")})}}},140:function(e,t,a){"use strict";Object.defineProperty(t,"__esModule",{value:!0});var s=a(69);t.default={name:"personalInfo",data:function(){return{name:"",address:"",age:"",job:"",sex:"Male"}},created:function(){},methods:{info_submit:function(e){var t=this;this.name=this.$refs.name.value,this.address=this.$refs.address.value,this.job=this.$refs.job.value,this.age=this.$refs.age.value;var a={name:this.name},i=(0,s.registerCheck)(a);i.then(function(e){var a=e.result.user_exist;if(a===!0&&t.$message({message:"Please choose another name, this one has been used by others",type:"error",offset:300}),t.checkEmpty()&&!a)return t.$router.push({name:"camera",query:{name:t.name,address:t.address,age:t.age,job:t.job,sex:t.sex}})})},checkEmpty:function(){return""===this.name?(this.$message({message:"Please enter your name",type:"error",offset:300}),!1):""===this.address?(this.$message({message:"Please enter your adress",type:"error",offset:300}),!1):""===this.age?(this.$message({message:"Please enter your age",type:"error",offset:300}),!1):""!==this.job||(this.$message({message:"Please enter your job",type:"error",offset:300}),!1)}}}},141:function(e,t){"use strict";Object.defineProperty(t,"__esModule",{value:!0}),t.default={name:"success"}},142:function(e,t,a){"use strict";function s(e){return e&&e.__esModule?e:{default:e}}Object.defineProperty(t,"__esModule",{value:!0});var i=a(70),n=s(i),r=a(71),o=s(r),c=a(121),d=s(c),u=d.default.create({baseURL:{NODE_ENV:"production"}.VUE_APP_BASE_API,timeout:5e3});u.interceptors.request.use(function(e){return e},function(e){console.log(e),o.default.reject(e)}),u.interceptors.response.use(function(e){var t=e.data;return 200!==t.code&&0!==t.code?o.default.reject(new Error(t.message||"Error")):t},function(e){var t="",a=JSON.parse((0,n.default)(e));if(a.response.status)switch(e.response.status){case 400:t="请求错误(400)，请重新申请";break;case 401:return t="登录错误(401)，请重新登录",(void 0).$router.replace("/login");case 403:t="拒绝访问(403)";break;case 404:t="请求出错(404)";break;case 408:t="请求超时(408)";break;case 500:t="服务器错误(500)，请重启软件或切换功能页！";break;case 501:t="服务未实现(501)";break;case 502:t="网络错误(502)";break;case 503:t="服务不可用(503)";break;case 504:t="网络超时(504)";break;case 505:t="HTTP版本不受支持(505)";break;default:t="网络连接出错"}else t="连接服务器失败,请退出重试!";return Message({showClose:!0,message:t,type:"error"}),o.default.reject(e)}),t.default=u},143:function(e,t){"use strict";!function(e,t){var a=e.documentElement,s="orientationchange"in window?"orientationchange":"resize",i=function(){var e=a.clientWidth;e&&(a.style.fontSize=20*(e/320)+"px")};e.addEventListener&&(t.addEventListener(s,i,!1),e.addEventListener("DOMContentLoaded",i,!1))}(document,window)},144:function(e,t,a){"use strict";function s(e){return e&&e.__esModule?e:{default:e}}Object.defineProperty(t,"__esModule",{value:!0});var i=a(234),n=s(i),r=a(235),o=s(r),c=a(236),d=s(c),u=a(237),_=s(u);t.default=[{path:"/",component:n.default,children:[{path:"",name:"personalInfo",component:d.default},{path:"/camera",name:"camera",component:o.default},{path:"/success",name:"success",component:_.default}]}]},145:function(e,t,a){"use strict";function s(e){return e&&e.__esModule?e:{default:e}}Object.defineProperty(t,"__esModule",{value:!0});var i=a(38);s(i);t.default={addNum:function(e,t){var a=e.commit,s=e.state;a("REMBER_ANSWER",t),s.itemNum<s.itemDetail.length&&a("ADD_ITEMNUM",1)},initializeData:function(e){var t=e.commit;t("INITIALIZE_DATA")}}},146:function(e,t,a){"use strict";function s(e){return e&&e.__esModule?e:{default:e}}Object.defineProperty(t,"__esModule",{value:!0});var i=a(5),n=s(i),r=a(244),o=s(r),c=a(147),d=s(c),u=a(145),_=s(u),l=a(38);s(l);n.default.use(o.default);var f={level:"第一周",itemNum:1,allTime:0,timer:"",itemDetail:[{topic_id:20,active_topic_id:4,type:"ONE",topic_name:"题目一",active_id:1,active_title:"欢乐星期五标题",active_topic_phase:"第一周",active_start_time:"1479139200",active_end_time:"1482163200",topic_answer:[{topic_answer_id:1,topic_id:20,answer_name:"答案aaaa",is_standard_answer:0},{topic_answer_id:2,topic_id:20,answer_name:"正确答案",is_standard_answer:0},{topic_answer_id:3,topic_id:20,answer_name:"答案cccc",is_standard_answer:0},{topic_answer_id:4,topic_id:20,answer_name:"答案dddd",is_standard_answer:1}]},{topic_id:21,active_topic_id:4,type:"MORE",topic_name:"题目二",active_id:1,active_title:"欢乐星期五标题",active_topic_phase:"第一周",active_start_time:"1479139200",active_end_time:"1482163200",topic_answer:[{topic_answer_id:5,topic_id:21,answer_name:"答案A",is_standard_answer:1},{topic_answer_id:6,topic_id:21,answer_name:"答案B",is_standard_answer:0},{topic_answer_id:7,topic_id:21,answer_name:"正确答案",is_standard_answer:0},{topic_answer_id:8,topic_id:21,answer_name:"答案D",is_standard_answer:0}]},{topic_id:21,active_topic_id:4,type:"MORE",topic_name:"题目三",active_id:1,active_title:"欢乐星期五标题",active_topic_phase:"第一周",active_start_time:"1479139200",active_end_time:"1482163200",topic_answer:[{topic_answer_id:9,topic_id:21,answer_name:"测试A",is_standard_answer:1},{topic_answer_id:10,topic_id:21,answer_name:"BBBBBB",is_standard_answer:0},{topic_answer_id:11,topic_id:21,answer_name:"CCCCCC",is_standard_answer:0},{topic_answer_id:12,topic_id:21,answer_name:"正确答案",is_standard_answer:0}]},{topic_id:21,active_topic_id:4,type:"MORE",topic_name:"题目四",active_id:1,active_title:"欢乐星期五标题",active_topic_phase:"第一周",active_start_time:"1479139200",active_end_time:"1482163200",topic_answer:[{topic_answer_id:13,topic_id:21,answer_name:"正确答案",is_standard_answer:1},{topic_answer_id:14,topic_id:21,answer_name:"A是错的",is_standard_answer:0},{topic_answer_id:15,topic_id:21,answer_name:"D是对的",is_standard_answer:0},{topic_answer_id:16,topic_id:21,answer_name:"C说的不对",is_standard_answer:0}]},{topic_id:21,active_topic_id:4,type:"MORE",topic_name:"题目五",active_id:1,active_title:"欢乐星期五标题",active_topic_phase:"第一周",active_start_time:"1479139200",active_end_time:"1482163200",topic_answer:[{topic_answer_id:17,topic_id:21,answer_name:"错误答案",is_standard_answer:1},{topic_answer_id:18,topic_id:21,answer_name:"正确答案",is_standard_answer:0},{topic_answer_id:19,topic_id:21,answer_name:"错误答案",is_standard_answer:0},{topic_answer_id:20,topic_id:21,answer_name:"错误答案",is_standard_answer:0}]}],answerid:[]};t.default=new o.default.Store({state:f,actions:_.default,mutations:d.default})},147:function(e,t,a){"use strict";function s(e){return e&&e.__esModule?e:{default:e}}Object.defineProperty(t,"__esModule",{value:!0});var i,n=a(153),r=s(n),o="ADD_ITEMNUM",c="REMBER_ANSWER",d="REMBER_TIME",u="INITIALIZE_DATA";t.default=(i={},(0,r.default)(i,o,function(e,t){e.itemNum+=t}),(0,r.default)(i,c,function(e,t){e.answerid.push(t)}),(0,r.default)(i,d,function(e){e.timer=setInterval(function(){e.allTime++},1e3)}),(0,r.default)(i,u,function(e){e.itemNum=1,e.allTime=0,e.answerid=[]}),i)},218:function(e,t){},219:function(e,t){},220:function(e,t){},221:function(e,t){},222:function(e,t){},223:function(e,t){},224:function(e,t,a){e.exports=a.p+"static/img/successful.jpg"},234:function(e,t,a){a(221);var s=a(37)(a(138),a(239),null,null);e.exports=s.exports},235:function(e,t,a){a(220);var s=a(37)(a(139),a(238),"data-v-1b5778f5",null);e.exports=s.exports},236:function(e,t,a){a(222);var s=a(37)(a(140),a(240),"data-v-2ea76344",null);e.exports=s.exports},237:function(e,t,a){a(223);var s=a(37)(a(141),a(241),"data-v-e83a5b3a",null);e.exports=s.exports},238:function(e,t){e.exports={render:function(){var e=this,t=e.$createElement,a=e._self._c||t;return a("div",[a("div",{staticStyle:{width:"370px"}},[e._m(0),e._v(" "),a("div",{staticClass:"input_wrap"},[a("video",{ref:"video",staticStyle:{"margin-top":"10px"},attrs:{width:"300",height:"240",autoplay:""}}),e._v(" "),a("canvas",{ref:"canvas",attrs:{width:"300",height:"240"}})])]),e._v(" "),a("div",{staticClass:"td_content"},[a("button",{staticClass:"info_button",on:{click:e.closeCamera}},[e._v("reshot")]),e._v(" "),a("button",{staticClass:"info_button",on:{click:e.photograph}},[e._v("shot")]),e._v(" "),a("button",{staticClass:"info_button",on:{click:e.submit}},[e._v("submit")])])])},staticRenderFns:[function(){var e=this,t=e.$createElement,a=e._self._c||t;return a("div",{staticClass:"selection_header"},[a("h1",{staticClass:"photo"},[e._v("Photo Registration")])])}]}},239:function(e,t){e.exports={render:function(){var e=this,t=e.$createElement,a=e._self._c||t;return a("div",[a("router-view")],1)},staticRenderFns:[]}},240:function(e,t){e.exports={render:function(){var e=this,t=e.$createElement,a=e._self._c||t;return a("div",[e._m(0),e._v(" "),a("div",{staticClass:"input_wrap"},[a("div",[a("form",{attrs:{id:"info_form"}},[a("table",[a("tr",[e._m(1),e._v(" "),a("td",[a("input",{ref:"name",staticClass:"input2",attrs:{width:"150px",type:"text"}})])]),e._v(" "),a("tr",[e._m(2),e._v(" "),a("td",[a("input",{ref:"address",staticClass:"input2",attrs:{width:"150px",type:"text"}})])]),e._v(" "),a("tr",[a("td",[a("span",{staticStyle:{width:"100px","font-size":"20px",height:"50px"}},[e._v(" Male*")]),e._v(" "),a("input",{directives:[{name:"model",rawName:"v-model",value:e.sex,expression:"sex"}],attrs:{type:"radio",value:"Male",name:"lg",checked:""},domProps:{checked:e._q(e.sex,"Male")},on:{change:function(t){e.sex="Male"}}})]),e._v(" "),a("td",[a("span",{staticStyle:{width:"100px","font-size":"20px",height:"50px"}},[e._v(" Female*")]),e._v(" "),a("input",{directives:[{name:"model",rawName:"v-model",value:e.sex,expression:"sex"}],attrs:{type:"radio",value:"Female",name:"lg"},domProps:{checked:e._q(e.sex,"Female")},on:{change:function(t){e.sex="Female"}}})])]),e._v(" "),a("tr",[e._m(3),e._v(" "),a("td",[a("input",{ref:"age",staticClass:"input2",attrs:{width:"150px",type:"text"}})])]),e._v(" "),a("tr",[e._m(4),e._v(" "),a("td",[a("input",{ref:"job",staticClass:"input2",attrs:{width:"150px",type:"text"}})])])])])])]),e._v(" "),a("div",{staticClass:"td_content"},[a("button",{attrs:{id:"info_button"},on:{click:e.info_submit}},[e._v("Next >")])])])},staticRenderFns:[function(){var e=this,t=e.$createElement,a=e._self._c||t;return a("div",{staticClass:"selection_header"},[a("h1",{staticStyle:{color:"#f7f7fd","font-size":"28px","margin-top":"50px","margin-left":"30px"}},[e._v("Basic Infomation")])])},function(){var e=this,t=e.$createElement,a=e._self._c||t;return a("td",[a("span",{staticStyle:{width:"100px","font-size":"20px",height:"50px"}},[e._v(" Name*")])])},function(){var e=this,t=e.$createElement,a=e._self._c||t;return a("td",[a("span",{staticStyle:{width:"100px","font-size":"20px",height:"50px"}},[e._v(" Address*")])])},function(){var e=this,t=e.$createElement,a=e._self._c||t;return a("td",[a("span",{staticStyle:{width:"100px","font-size":"20px",height:"50px"}},[e._v(" Age*")])])},function(){var e=this,t=e.$createElement,a=e._self._c||t;return a("td",[a("span",{staticStyle:{width:"100px","font-size":"20px",height:"50px"}},[e._v(" Job*")])])}]}},241:function(e,t,a){e.exports={render:function(){var e=this,t=e.$createElement;e._self._c||t;return e._m(0)},staticRenderFns:[function(){var e=this,t=e.$createElement,s=e._self._c||t;return s("div",{staticClass:"main"},[s("img",{attrs:{src:a(224)}}),e._v(" "),s("span",{staticClass:"hint"},[e._v("Registration Success!")])])}]}}});
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 25 10:20:23 2019
@author: WellenWoo
模型选择窗口,训练超参数设置窗口;
"""
import wx

class ProgressStatusBar(wx.StatusBar):
    """带有进度条的状态栏;"""
    def __init__(self, parent, id_ = wx.ID_ANY, 
                 style = wx.SB_FLAT, name = "ProgressStatusBar"):
      super(ProgressStatusBar, self).__init__(parent, id_, style, name)
      
      self.prog = wx.Gauge(self, style = wx.GA_HORIZONTAL)
      self.prog.Hide()
      
      self.busy = False
      
      self.SetFieldsCount(number = 2, widths = [-1, 155] ) #状态栏分为两半
              
    def work(self):
        if self.busy:
            self.prog.Show()
            lfield = self.GetFieldsCount() - 1
            rect = self.GetFieldRect(lfield)
            prog_pos = (rect.x + 2, rect.y + 0)
            self.prog.SetPosition(prog_pos)
            prog_size = (rect.width - 8, rect.height - 0)
            self.prog.SetSize(prog_size)
            self.prog.SetValue(90)            
        else:
            self.prog.SetValue(0)
            self.prog.Hide()
                    
class ModelPanel(wx.Panel):
    """独立窗口的选择框;"""
    def __init__(self, parent):
        super(ModelPanel, self).__init__(parent)
        
        static_font = wx.Font(12, wx.SWISS, wx.NORMAL, wx.NORMAL)
        
        label = wx.StaticText(self, -1, label = "Model")
        label.SetBackgroundColour(wx.Colour(200, 200, 200))
        label.SetFont(static_font)
        
        items = ["squeezenet","densenet",  "alexNet", "vgg"]
        self.choice = wx.Choice(self, choices = items)
        self.choice.SetSelection(0)
        
        sizer = wx.BoxSizer()
        
        sizer.Add(label, 1, wx.EXPAND | wx.ALL, 10)
        sizer.Add(self.choice, 1, wx.EXPAND | wx.ALL, 10)
        self.SetSizer(sizer)
        
        self.Bind(wx.EVT_CHOICE, self.OnChoice)
        
    def OnChoice(self, evt):
        app = wx.GetApp()
        frame = app.GetTopWindow()
        
        sel = self.choice.GetSelection()
        if sel == 0:            
            val = r"../models/squeezenet1_0-a815701f.pth"
        elif sel == 1:
            val = r"../models/densenet121-a639ec97.pth"
        elif sel == 2:
            val = r"../models/alexnet-owt-4df8aa71.pth"
        elif sel == 3:
            val = r"../models/vgg19-dcbb9e9d.pth"
        frame.model_path = val
        
class LangDialog(wx.Dialog):
    """设置语言的弹窗选择框;"""
    def __init__(self, *arg, **kw):
        # grab the config keyword and remove it from the dict
        self.lang_config = kw["config"]
        del kw['config']
        
        wx.Dialog.__init__(self, *arg, **kw)
                
        self.SetTitle("Language Setting")
        self.Bind(wx.EVT_CHOICE, self.OnChoice)
        self.__layout()
        
#        app = wx.GetApp()
#        self.frame = app.GetTopWindow()
        
    def __layout(self):
        items = ["English","简体中文"]
        
        self.label = wx.StaticText(self, -1, "Language")
        self.choice = wx.Choice(self, choices = items)
        self.choice.SetSelection(0)
        
        self.btn_ok = wx.Button(self, wx.ID_OK, "")
        self.btn_cancel = wx.Button(self, wx.ID_CANCEL, "")
        
        sizer0 = wx.BoxSizer(wx.HORIZONTAL)
        sizer1 = wx.FlexGridSizer(2, 5, 5)
        
        sizer1.Add(self.label, 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 4)
        sizer1.Add(self.choice, 0, wx.EXPAND, 4)
        sizer1.Add(self.btn_ok, 0, 0, 0)
        sizer1.Add(self.btn_cancel, 0, 0, 0)
        
        sizer0.Add(sizer1, 0, wx.ALL | wx.ALIGN_BOTTOM, 4)
        
        self.SetSizer(sizer0)
        sizer0.Fit(self)
        self.Layout()

    def OnChoice(self, evt):
        sel = self.choice.GetSelection()
        if sel == 0:            
            val = "LANGUAGE_ENGLISH"
        elif sel == 1:
            val = "LANGUAGE_CHINESE_SIMPLIFIED" #简体中文
        self.lang_config.Write('lang', val) 
        
class HyperDailog(wx.Dialog):
    """训练超参数的选择对话框;"""
    def __init__(self, *arg, **kw):
        wx.Dialog.__init__(self, *arg, **kw)
        
        self.__DoLayout()
        self.__Binds()
        self.Show()
        
    def __DoLayout(self):
        static_font = wx.Font(12, wx.SWISS, wx.NORMAL, wx.NORMAL)
        
        lb1 = wx.StaticText(self, -1, label = "num steps")
        lb2 = wx.StaticText(self, -1, label = "style weight")
        lb3 = wx.StaticText(self, -1, label = "content weight")
        
        self.in1 = wx.TextCtrl(self, -1, value = "300")
        self.in2 = wx.TextCtrl(self, -1, value = "1000")
        self.in3 = wx.TextCtrl(self, -1, value = "1")
        
        self.btn_ok = wx.Button(self, wx.ID_OK, "")
        self.btn_cancel = wx.Button(self, wx.ID_CANCEL, "")
        
        for i in [lb1, lb2, lb3, self.in1, self.in2, self.in3]:
            i.SetFont(static_font)
            i.SetBackgroundColour(wx.Colour(200, 200, 200))
            
        sizer0 = wx.FlexGridSizer(cols = 2, vgap = 5, hgap  = 5)
        
        sizer0.Add(lb1, 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 4)
        sizer0.Add(self.in1, 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 4)
        sizer0.Add(lb2, 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 4)
        sizer0.Add(self.in2, 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 4)
        sizer0.Add(lb3, 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 4)
        sizer0.Add(self.in3, 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 4)
        sizer0.Add(self.btn_ok, 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 4)
        sizer0.Add(self.btn_cancel, 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 4)
        
        self.SetSizer(sizer0)
        sizer0.Fit(self)
        self.Layout()
        
    def __Binds(self):
        self.Bind(wx.EVT_BUTTON, self.OnOk, self.btn_ok)
        self.Bind(wx.EVT_BUTTON, self.OnCancel, self.btn_cancel)
        self.Bind(wx.EVT_CLOSE, self.OnCloseWindow)
        
    def OnOk(self, evt):
        app = wx.GetApp()
        frame = app.GetTopWindow()

        frame.num_steps = self.in1.GetValue()
        frame.style_weight = self.in2.GetValue()
        frame.content_weight = self.in3.GetValue()
        self.Close()
            
    def OnCancel(self, evt):
        self.Close()
        
    def OnCloseWindow(self, evt):
        self.Destroy()
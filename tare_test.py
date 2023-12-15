from addtype import turnold2new
import sys
import os
import javalang
# from ast import nodes
from graphviz import Digraph
import json
import pickle
from tqdm import tqdm
import numpy as np
from tare_runvalid import *
from stringfycode import stringfyRoot
from copy import deepcopy
import time
import io
import subprocess
from Searchnode import Node, NodeWithType

linenode = ['Statement_ter', 'BreakStatement_ter', 'ReturnStatement_ter', 'ContinueStatement', 'ContinueStatement_ter', 'LocalVariableDeclaration', 'condition', 'control',
            'BreakStatement', 'ContinueStatement', 'ReturnStatement', "parameters", 'StatementExpression', 'return_type']


# os.environ["CUDA_VISIBLE_DEVICES"]="1, 4"
def getLocVar(node):
    varnames = []
    if node.name == 'VariableDeclarator':
        currnode = -1
        for x in node.child:
            if x.name == 'name':
                currnode = x
                break
        varnames.append((currnode.child[0].name, node))
    if node.name == 'FormalParameter':
        currnode = -1
        for x in node.child:
            if x.name == 'name':
                currnode = x
                break
        varnames.append((currnode.child[0].name, node))
    if node.name == 'InferredFormalParameter':
        currnode = -1
        for x in node.child:
            if x.name == 'name':
                currnode = x
                break
        varnames.append((currnode.child[0].name, node))
    for x in node.child:
        varnames.extend(getLocVar(x))
    return varnames


n = 0


def setid(root):
    global n
    root.id = n
    n += 1
    for x in root.child:
        setid(x)


def setSameid(root, rootsave):
    rootsave.id = root.id
    for i in range(len(root.child)):
        setSameid(root.child[i], rootsave.child[i])
    return


def checkvar(root):
    ans = []
    if len(root.child) == 0:
        ans.append(root.name)
    for x in root.child:
        ans.extend(checkvar(x))
    return ans


def solveLongTree(root, subroot, currnode):
    global n
    m = 'None'
    troot = 'None'
    for x in root.child:
        if x.name == 'name':
            m = x.child[0].name
    if len(root.getTreestr().strip().split()) >= 1000:
        tmp = subroot
        if len(tmp.getTreestr().split()) >= 1000:
            assert 0
        lasttmp = None
        while True:
            if len(tmp.getTreestr().split()) >= 1000:
                break
            lasttmp = tmp
            tmp = tmp.father
        index = tmp.child.index(lasttmp)
        ansroot = NodeWithType(tmp.nameo, 0)
        ansroot.child.append(lasttmp)
        ansroot.num = 2 + len(lasttmp.getTreestr().strip().split())
        while True:
            b = True
            afternode = tmp.child.index(ansroot.child[-1]) + 1
            if afternode < len(tmp.child) and ansroot.num + tmp.child[afternode].getNum() < 1000:
                b = False
                ansroot.child.append(tmp.child[afternode])
                ansroot.num += tmp.child[afternode].getNum()
            prenode = tmp.child.index(ansroot.child[0]) - 1
            if prenode >= 0 and ansroot.num + tmp.child[prenode].getNum() < 1000:
                b = False
                ansroot.child = [tmp.child[prenode]] + ansroot.child
                ansroot.num += tmp.child[prenode].getNum()
            if b:
                break
        troot = ansroot
    else:
        troot = root
    n = 0
    setid(troot)
    varnames = getLocVar(root)
    ans = checkvar(troot)

    tmp = []
    for var in varnames:
        if var[0] in ans:
            tmp.append(var)
    varnames = tmp
    fnum = -1
    vnum = -1
    vardic = {}
    vardic[m] = 'meth0'
    varwithts = [('meth0', 'ptype', 1, '[maskt]')]
    if currnode is None:
        curridm = 1e10
    else:
        curridm = currnode.id
    typedic = {}
    for x in varnames:
        if x[0] in vardic:
            continue
        if x[1].name == 'VariableDeclarator':
            vnum += 1
            vardic[x[0]] = 'loc' + str(vnum)
            t = -1
            for s in x[1].father.father.child:
                if s.name == 'type':
                    if len(s.child) == 1:  # normal type
                        t = s.child[0].child[0].child[0].name
                        break
                    elif len(s.child) == 2:  # array type
                        t = s.child[0].child[0].child[0].name + '[]'
                        break
                    else:
                        assert 0
            if t == -1:
                t = 'ptype'
            flag = 0
            if x[1].id < curridm:
                flag = 1
            varwithts.append((vardic[x[0]], gettype(t), flag, t))
            typedic[x[0]] = t[:-4]
        else:
            fnum += 1
            vardic[x[0]] = 'par' + str(fnum)
            t = -1
            for s in x[1].child:
                if s.name == 'type':
                    if len(s.child) == 1:
                        t = s.child[0].child[0].child[0].name
                        break
                    elif len(s.child) == 2:
                        t = s.child[0].child[0].child[0].name + '[]'
                        break
            if t == -1:
                t = 'ptype'
            flag = 0
            if x[1].id < curridm:
                flag = 1
            varwithts.append((vardic[x[0]], gettype(t), flag, t))
            typedic[x[0]] = t[:-4]
    # print(varwithts)
    return troot, vardic, typedic, varwithts


def addter(root):
    if len(root.child) == 0:
        root.name += "_ter"
        root.namewithouttype = root.name
    for x in root.child:
        addter(x)
    return


def setProb(r, p):
    r.possibility = p  # max(min(np.random.normal(0.8, 0.1, 10)[0], 1), 0)
    for x in r.child:
        setProb(x, p)


def setProbwithPre(r, prob):
    # max(min(np.random.normal(0.8, 0.1, 10)[0], 1), 0)
    r.possibility = prob[r.id]
    for x in r.child:
        setProbwithPre(x, prob)


def getLineNode(root, block, add=True):
    ans = []
    block = block + root.name
    # print(root.name, 'lll')
    for x in root.child:
        if x.name in linenode:
            if 'info' in x.getTreestr() or 'assert' in x.getTreestr() or 'logger' in x.getTreestr() or 'LOGGER' in x.getTreestr() or 'system.out' in x.getTreestr().lower():
                continue
            x.block = block
            ans.append(x)
        else:
            # print(x.name)
            s = ""
            if not add:
                s = block
                # tmp = getLineNode(x, block)
            else:
                s = block + root.name
            # print(block + root.name + "--------")
            tmp = getLineNode(x, block)
            '''if x.name == 'then_statement' and tmp == []:
        print(tmp)
        print(x.father.printTree(x.father))
        assert(0)'''
            ans.extend(tmp)
    return ans


def getroottree(tokens, isex=False):
    if isinstance(tokens[0], tuple):
        root = Node(tokens[0][0], 0)
        root.position = tokens[0][1]
    else:
        root = Node(tokens[0], 0)
    currnode = root
    idx = 1
    for i, x in enumerate(tokens[1:]):
        if x != "^":
            if isinstance(x, tuple):
                nnode = Node(x[0], idx)
                nnode.position = x[1]
            else:
                nnode = Node(x, idx)
            nnode.father = currnode
            currnode.child.append(nnode)
            currnode = nnode
            idx += 1
        else:
            currnode = currnode.father
    return root


def getroottree_with_type(tokens, isex=False):
    if isinstance(tokens[0], tuple):
        root = NodeWithType(tokens[0][0], 0)
        root.position = tokens[0][1]
    else:
        root = NodeWithType(tokens[0], 0)
    currnode = root
    idx = 1
    for i, x in enumerate(tokens[1:]):
        if x != "^":
            if isinstance(x, tuple):
                nnode = NodeWithType(x[0], idx)
                nnode.position = x[1]
            else:
                nnode = NodeWithType(x, idx)
            nnode.father = currnode
            currnode.child.append(nnode)
            currnode = nnode
            idx += 1
        else:
            currnode = currnode.father
    return root


def ismatch(root, subroot):
    index = 0
    # assert(len(subroot.child) <= len(root.child))
    # print(len(subroot.child), len(root.child))
    for x in subroot.child:
        while index < len(root.child) and root.child[index].name != x.name:
            index += 1
        if index == len(root.child):
            return False
        if not ismatch(root.child[index], x):
            return False
        index += 1
    return True


def findSubtree(root, subroot):
    if root.name == subroot.name:
        if ismatch(root, subroot):
            return root
    for x in root.child:
        tmp = findSubtree(x, subroot)
        if tmp:
            return tmp
    return None


def generateAST(tree):
    sub = []
    if not tree:
        return ['None', '^']
    if isinstance(tree, str):
        tmpStr = tree
        tmpStr = tmpStr.replace(" ", "").replace(":", "")
        if "\t" in tmpStr or "'" in tmpStr or "\"" in tmpStr:
            tmpStr = "<string>"
        if len(tmpStr) == 0:
            tmpStr = "<empty>"
        if tmpStr[-1] == "^":
            tmpStr += "<>"
        sub.append(tmpStr)
        sub.append("^")
        return sub
    if isinstance(tree, list):
        if len(tree) == 0:
            sub.append("empty")
            sub.append("^")
        else:
            for ch in tree:
                subtree = generateAST(ch)
                sub.extend(subtree)
        return sub
    position = None
    if hasattr(tree, 'position'):
        # assert(0)
        position = tree.position
    curr = type(tree).__name__
    # print(curr)
    if True:
        if False:
            assert 0  # sub.append((str(getLiteral(tree.children)))
        else:
            sub.append((curr, position))
            try:
                for x in tree.attrs:
                    if x == "documentation":
                        continue
                    if not getattr(tree, x):
                        continue
                    '''if x == 'prefix_operators':
                        node = getattr(tree, x)
                        print(type(node))
                        print(len(node))
                        print(node[0])
                        assert(0)
                    if type(getattr(tree, x)).__name__ not in nodes:
                        print(type(getattr(tree, x)).__name__)
                        continue'''
                    sub.append(x)
                    node = getattr(tree, x)
                    if isinstance(node, list):
                        if len(node) == 0:
                            sub.append("empty")
                            sub.append("^")
                        else:
                            for ch in node:
                                subtree = generateAST(ch)
                                sub.extend(subtree)
                    elif isinstance(node, javalang.tree.Node):
                        subtree = generateAST(node)
                        sub.extend(subtree)
                    elif not node:
                        continue
                    elif isinstance(node, str):
                        tmpStr = node
                        tmpStr = tmpStr.replace(" ", "").replace(":", "")
                        if "\t" in tmpStr or "'" in tmpStr or "\"" in tmpStr:
                            tmpStr = "<string>"
                        if len(tmpStr) == 0:
                            tmpStr = "<empty>"
                        if tmpStr[-1] == "^":
                            tmpStr += "<>"
                        sub.append(tmpStr)
                        sub.append("^")
                    elif isinstance(node, set):
                        for ch in node:
                            subtree = generateAST(ch)
                            sub.extend(subtree)
                    elif isinstance(node, bool):
                        sub.append(str(node))
                        sub.append("^")
                    else:
                        print(type(node))
                        assert 0
                    sub.append("^")
            except AttributeError:
                assert 0
                pass
        sub.append('^')
        return sub
    else:
        print(curr)
    return sub


'''def setProb(root, subroot, prob):
    root.possibility = max(min(max(root.possibility, prob), 0.98), 0.01)
    index = 0
    assert(len(subroot.child) <= len(root.child))
    #print(len(subroot.child), len(root.child))
    for x in subroot.child:
        while root.child[index].name != x.name:
            #print(root.child[index].name, x.name)
            index += 1
        setProb(root.child[index], x, prob)
        index += 1'''


def getSubroot(treeroot):
    currnode = treeroot
    lnode = None
    mnode = None
    while currnode:
        if currnode.name in linenode:
            lnode = currnode
            break
        currnode = currnode.father
    currnode = treeroot
    while currnode:
        if currnode.name == 'MethodDeclaration' or currnode.name == 'ConstructorDeclaration':
            mnode = currnode
            break
        currnode = currnode.father
    return lnode, mnode


def getNodeById(root, line):
    if root.position:
        if root.position.line == line and root.name != 'IfStatement' and root.name != 'ForStatement' and root.name != 'WhileStatement':
            return root
    for x in root.child:
        t = getNodeById(x, line)
        if t:
            return t
    return None


def getById(root, idx):
    if root.id == idx:
        return root
    for x in root.child:
        t = getById(x, idx)
        if t is not None:
            return t
    return None


curridx = 0


def getNodeNo(root, subroot, idx):
    global curridx
    if root.id == subroot.id:
        return curridx
    curridx += 1
    for x in root.child:
        t = getNodeNo(x, subroot, idx)
        if t != -1:
            return t
    return -1


def containID(root):
    ans = []
    if root.position is not None:
        ans.extend([root.position.line])
    for x in root.child:
        ans.extend(containID(x))
    return ans


def getAssignMent(root):
    if root.name == 'Assignment':
        return root
    for x in root.child:
        t = getAssignMent(x)
        if t:
            return t
    return None


def isAssign(line):
    # sprint(4, line.getTreestr())
    if 'Assignment' not in line.getTreestr():
        return False
    anode = getAssignMent(line)
    if anode.child[0].child[0].name == 'MemberReference' and anode.child[1].child[0].name == 'MethodInvocation':
        try:
            m = anode.child[0].child[0].child[0].child[0].name
            v = anode.child[1].child[0].child[0].child[0].name
        except:
            return False
        # print(m, v)
        return m == v
    if anode.child[0].child[0].name == 'MemberReference':
        try:
            m = anode.child[0].child[0].child[0].child[0].name
        except:
            return False
        if "qualifier " + m in anode.child[1].getTreestr():
            return True
    return False


def gettype(t):
    if t in ['int_ter', 'double_ter', 'long_ter']:
        return 'numeric'
    elif t in ['boolean_ter']:
        return 'bool'
    elif 'String' in t:
        return 'string'
    else:
        return 'ptype'


model = test()

buggy_folder_path = 'Valid/buggy_methods'
meta_folder_path = 'Valid/metas'
meta_file_list = os.listdir(meta_folder_path)

for meta_file in tqdm(meta_file_list):
    lineid = 0
    classname = ""
    idss = meta_file.split('.')[0]
    # meta data file
    with open(os.path.join(meta_folder_path, meta_file), 'r', encoding='UTF-8') as f1:
        meta = f1.readline()
        classname = meta.split('<sep>')[4].split('@')[0].split('\\')[-1].split('.')[0]  # 去掉.java后缀
        buggy_lines = meta.split('<sep>')[2]
        lineid = eval(buggy_lines.replace('[', '').replace(']', '').split(':')[0]) + 1

    location = []
    prob = 0
    location.append((classname, prob, eval(lineid)))
    locationdict = {}
    data = []
    for j in range(300):
        if j >= len(location):
            break
        try:
            patchdict = {}
            ac = location[j]
            classname = ac[0]
            if '$' in classname:
                classname = classname[:classname.index('$')]
            s = classname
            # buggy code file: meta_file == buggy_file
            filepath = os.path.join(buggy_folder_path, meta_file)
            # 只有method的代码需要外嵌入一个class语句才能通过编译
            lines1 = ""
            liness = []
            line_list = open(filepath, "r", encoding='UTF-8').readlines()
            spaces_before = len(line_list[0]) - len(line_list[0].lstrip()) - 4
            lines1 += " " * spaces_before + "public class " + classname + " {" + "\n"
            lineid += 1
            liness.append(" " * spaces_before + "public class " + classname + " {" + "\n")
            for line in line_list:
                lines1 += line
                liness.append(line)
            lines1 += " " * spaces_before + "}" + "\n"
            liness.append(" " * spaces_before + "}" + "\n")

            tokens = javalang.tokenizer.tokenize(lines1)
            parser = javalang.parser.Parser(tokens)
            tree = parser.parse()
            tmproot = getroottree(generateAST(tree))
            lineid = ac[2]

            currroot = getNodeById(tmproot, lineid)
            lnode, mnode = getSubroot(currroot)
            if mnode is None:
                continue
            oldcode = liness[ac[2] - 1]
            isIf = True
            subroot = lnode
            treeroot = mnode
            presubroot = None
            aftersubroot = None
            linenodes = getLineNode(treeroot, "")
            if subroot not in linenodes:
                continue
            currid = linenodes.index(subroot)
            if currid > 0:
                presubroot = linenodes[currid - 1]
            if currid < len(linenodes) - 1:
                aftersubroot = linenodes[currid + 1]
            setProb(treeroot, 2)
            if subroot is None:
                continue
            if True:
                cid = set(containID(subroot))
                subrootsave = subroot
                setProb(treeroot, 2)
                if subroot is not None:
                    setProb(subroot, 1)
                if aftersubroot is not None:
                    setProb(aftersubroot, 4)
                if presubroot is not None:
                    setProb(presubroot, 3)
                curridx = 0
                getNodeNo(treeroot, subroot, 0)
                subidx = curridx
                prob = subroot.getTreeProb(treeroot)
                newTree = turnold2new(subroot.printTree(treeroot).split())
                for pj in range(len(newTree) - 1):
                    if newTree[pj + 1] == '^' and newTree[pj] != '^':
                        newTree[pj] = newTree[pj] + '_ter'
                addter(subrootsave)
                treeroot = getroottree_with_type(newTree)
                setProbwithPre(treeroot, prob)
                subroot = getById(treeroot, subidx)
                linenodes = getLineNode(treeroot, "")
                idx = linenodes.index(subroot)
                if idx + 1 < len(linenodes):
                    curridm = linenodes[idx + 1]
                else:
                    curridm = None

                maxl = -1
                minl = 1e10
                for l in cid:
                    maxl = max(maxl, l - 1)
                    minl = min(minl, l - 1)
                precode = (0, minl)
                aftercode = (maxl + 1, len(liness))
                oldcode = (minl, maxl + 1)
                troot, vardic, typedic, varwithts = solveLongTree(treeroot, subroot, curridm)
                setSameid(subroot, subrootsave)
                data.append(
                    {'treeroot': treeroot, 'troot': troot, 'oldcode': oldcode, 'filepath': filepath, 'subroot': subrootsave, 'vardic': vardic, 'typedic': typedic, 'idss': idss,
                     'classname': classname, 'precode': precode, 'aftercode': aftercode, 'tree': troot.printTreeWithVar(troot, vardic), 'prob': troot.getTreeProb(troot), 'mode': 0,
                     'line': lineid, 'isa': False, 'varwithtype': varwithts, 'extended': j < 5})
        except:
            print('data error: ', location[j])
            continue

    if not os.path.exists('generation/Valid'):
        os.mkdir('generation/Valid')
    solveone(data, model, 'generation/Valid')
